/*
 * CPython binding of the modeFinder C library (csrc/).  Exposes
 *
 *   _emp_mode(data, fix_lower, fix_upper, smooth, smooth_option,
 *             smooth_density, density_points, m_degree, sample_size,
 *             m_knots) -> dict
 *
 * which modefinder.emp_mode() wraps.  data may be any object exporting a
 * contiguous float64 buffer (numpy array, array.array('d'), ...) or any
 * sequence of numbers.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "modefinder.h"

/* Copy data into a freshly malloc'ed double array. */
static double *as_doubles(PyObject *obj, Py_ssize_t *n)
{
    double *out;
    Py_buffer view;

    if (PyObject_CheckBuffer(obj) &&
        PyObject_GetBuffer(obj, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        if (view.format && strcmp(view.format, "d") == 0 &&
            view.itemsize == sizeof(double)) {
            *n = view.len / (Py_ssize_t) sizeof(double);
            out = (double *) PyMem_Malloc((*n > 0 ? *n : 1) * sizeof(double));
            if (!out) {
                PyBuffer_Release(&view);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(out, view.buf, *n * sizeof(double));
            PyBuffer_Release(&view);
            return out;
        }
        PyBuffer_Release(&view);
    }
    PyErr_Clear();

    PyObject *seq = PySequence_Fast(obj, "data must be a sequence of numbers");
    if (!seq) return NULL;
    *n = PySequence_Fast_GET_SIZE(seq);
    out = (double *) PyMem_Malloc((*n > 0 ? *n : 1) * sizeof(double));
    if (!out) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < *n; i++) {
        out[i] = PyFloat_AsDouble(items[i]);
        if (out[i] == -1.0 && PyErr_Occurred()) {
            Py_DECREF(seq);
            PyMem_Free(out);
            return NULL;
        }
    }
    Py_DECREF(seq);
    return out;
}

static PyObject *doubles_to_list(const double *x, int n)
{
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *v = PyFloat_FromDouble(x[i]);
        if (!v) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, v);
    }
    return list;
}

/* d[key] = value, stealing the reference to value */
static int set_item(PyObject *d, const char *key, PyObject *value)
{
    if (!value) return -1;
    int rc = PyDict_SetItemString(d, key, value);
    Py_DECREF(value);
    return rc;
}

static PyObject *result_to_dict(const mf_result *r)
{
    PyObject *d = PyDict_New();
    if (!d) return NULL;
    static const char *methods[] = {"Bernstein", "kernel", "empirical"};

    if (set_item(d, "method", PyUnicode_FromString(methods[r->method])) < 0 ||
        set_item(d, "mode", PyFloat_FromDouble(r->mode)) < 0 ||
        set_item(d, "message", PyUnicode_FromString(r->message)) < 0)
        goto fail;

    if (r->method == MF_METHOD_BERNSTEIN) {
        if (set_item(d, "degree_chosen", PyBool_FromLong(r->degree_chosen)) < 0 ||
            set_item(d, "degree", PyLong_FromLong(r->degree)) < 0 ||
            set_item(d, "alpha", PyFloat_FromDouble(r->alpha)) < 0 ||
            set_item(d, "p_value", PyFloat_FromDouble(r->p_value)) < 0 ||
            set_item(d, "max_point", PyFloat_FromDouble(r->max_point)) < 0 ||
            set_item(d, "prob_mid", PyFloat_FromDouble(r->prob_mid)) < 0 ||
            set_item(d, "prob_left", PyFloat_FromDouble(r->prob_left)) < 0 ||
            set_item(d, "prob_right", PyFloat_FromDouble(r->prob_right)) < 0)
            goto fail;
    } else if (r->method == MF_METHOD_EMPIRICAL) {
        if (set_item(d, "knots", PyLong_FromLong(r->knots)) < 0)
            goto fail;
    }

    if (r->density_x) {
        if (set_item(d, "x", doubles_to_list(r->density_x, r->n_density)) < 0 ||
            set_item(d, "y", doubles_to_list(r->density_y, r->n_density)) < 0)
            goto fail;
    }
    return d;

fail:
    Py_DECREF(d);
    return NULL;
}

/* None -> *has = 0; otherwise a float */
static int opt_double(PyObject *obj, int *has, double *value)
{
    if (obj == Py_None) {
        *has = 0;
        return 0;
    }
    *value = PyFloat_AsDouble(obj);
    if (*value == -1.0 && PyErr_Occurred()) return -1;
    *has = 1;
    return 0;
}

/* None -> 0 (automatic); otherwise an int */
static int opt_int(PyObject *obj, const char *name, int *value)
{
    if (obj == Py_None) {
        *value = 0;
        return 0;
    }
    long v = PyLong_AsLong(obj);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (v < 1 || v > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s must be a positive integer", name);
        return -1;
    }
    *value = (int) v;
    return 0;
}

static PyObject *py_emp_mode(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", "fix_lower", "fix_upper", "smooth",
                             "smooth_option", "smooth_density",
                             "density_points", "m_degree", "sample_size",
                             "m_knots", NULL};
    PyObject *data_obj, *lower_obj = Py_None, *upper_obj = Py_None;
    PyObject *m_degree_obj = Py_None, *m_knots_obj = Py_None;
    int smooth = 1, smooth_density = 0, density_points = 512, sample_size = 1000;
    const char *smooth_option = "Bernstein";
    mf_options opt;
    mf_result res;
    Py_ssize_t n;
    double *data;
    int status;
    (void) self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOpspiOiO", kwlist,
                                     &data_obj, &lower_obj, &upper_obj,
                                     &smooth, &smooth_option, &smooth_density,
                                     &density_points, &m_degree_obj,
                                     &sample_size, &m_knots_obj))
        return NULL;

    mf_default_options(&opt);
    opt.smooth = smooth;
    opt.smooth_density = smooth_density;
    opt.density_points = density_points;
    opt.sample_size = sample_size;
    if (strcmp(smooth_option, "Bernstein") == 0)
        opt.smooth_option = MF_SMOOTH_BERNSTEIN;
    else if (strcmp(smooth_option, "kernel") == 0)
        opt.smooth_option = MF_SMOOTH_KERNEL;
    else {
        PyErr_SetString(PyExc_ValueError,
                        "smooth_option must be \"Bernstein\" or \"kernel\"");
        return NULL;
    }
    if (opt_double(lower_obj, &opt.has_fix_lower, &opt.fix_lower) < 0 ||
        opt_double(upper_obj, &opt.has_fix_upper, &opt.fix_upper) < 0 ||
        opt_int(m_degree_obj, "m_degree", &opt.m_degree) < 0 ||
        opt_int(m_knots_obj, "m_knots", &opt.m_knots) < 0)
        return NULL;
    if (smooth && opt.smooth_option == MF_SMOOTH_BERNSTEIN &&
        m_degree_obj != Py_None && opt.m_degree < 2) {
        PyErr_SetString(PyExc_ValueError, "m_degree must be at least 2");
        return NULL;
    }

    data = as_doubles(data_obj, &n);
    if (!data) return NULL;

    Py_BEGIN_ALLOW_THREADS
    status = mf_emp_mode(data, (size_t) n, &opt, &res);
    Py_END_ALLOW_THREADS
    PyMem_Free(data);

    if (status != MF_OK) {
        if (status == MF_ERR_NOMEM) return PyErr_NoMemory();
        PyErr_SetString(status == MF_ERR_NUMERIC ? PyExc_ArithmeticError
                                                 : PyExc_ValueError,
                        mf_strerror(status));
        return NULL;
    }
    PyObject *out = result_to_dict(&res);
    mf_result_free(&res);
    return out;
}

static PyMethodDef methods[] = {
    {"_emp_mode", (PyCFunction)(void (*)(void)) py_emp_mode,
     METH_VARARGS | METH_KEYWORDS,
     "Low-level mode estimator; use modefinder.emp_mode() instead."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_modefinder",
    "C implementation of the modeFinder mode estimator.", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__modefinder(void)
{
    return PyModule_Create(&moduledef);
}
