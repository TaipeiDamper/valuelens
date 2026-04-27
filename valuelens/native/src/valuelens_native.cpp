#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

inline uint8_t clamp_u8(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return static_cast<uint8_t>(v);
}

std::vector<uint8_t> build_quant_lut(int levels, int min_value, int max_value, double exp_value) {
    std::vector<uint8_t> lut(256, 0);
    const int denom = std::max(1, max_value - min_value);
    const double gamma = std::pow(2.0, exp_value);

    for (int i = 0; i < 256; ++i) {
        double norm = static_cast<double>(i - min_value) / static_cast<double>(denom);
        norm = std::clamp(norm, 0.0, 1.0);
        if (exp_value != 0.0) {
            norm = std::pow(norm, gamma);
        }
        int idx = static_cast<int>(std::floor(norm * levels));
        idx = std::clamp(idx, 0, levels - 1);
        lut[static_cast<size_t>(i)] = static_cast<uint8_t>(idx);
    }
    return lut;
}

std::vector<uint8_t> build_display_lut(int levels, int display_min, int display_max, double display_exp) {
    std::vector<uint8_t> lut(static_cast<size_t>(levels), 0);
    const double gamma = std::pow(2.0, display_exp);
    for (int i = 0; i < levels; ++i) {
        double norm = (levels <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(levels - 1);
        if (display_exp != 0.0) {
            norm = std::pow(norm, gamma);
        }
        const double mapped = norm * static_cast<double>(display_max - display_min) + static_cast<double>(display_min);
        lut[static_cast<size_t>(i)] = clamp_u8(static_cast<int>(std::lround(mapped)));
    }
    return lut;
}

py::tuple quantize_fast(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bgr,
    int levels,
    int min_value,
    int max_value,
    double exp_value,
    int display_min,
    int display_max,
    double display_exp
) {
    if (levels < 2) {
        throw std::invalid_argument("levels must be >= 2");
    }
    if (min_value < 0 || min_value > 255 || max_value < 0 || max_value > 255) {
        throw std::invalid_argument("min/max values must be in [0, 255]");
    }
    if (display_min < 0 || display_min > 255 || display_max < 0 || display_max > 255) {
        throw std::invalid_argument("display min/max values must be in [0, 255]");
    }
    if (max_value <= min_value) {
        max_value = min_value + 1;
    }
    if (display_max < display_min) {
        std::swap(display_max, display_min);
    }

    auto in = bgr.unchecked<3>();
    if (in.shape(2) < 3) {
        throw std::invalid_argument("bgr input must have 3 channels");
    }
    const py::ssize_t h = in.shape(0);
    const py::ssize_t w = in.shape(1);

    auto quant_lut = build_quant_lut(levels, min_value, max_value, exp_value);
    auto display_lut = build_display_lut(levels, display_min, display_max, display_exp);

    py::array_t<uint8_t> out_gray(py::array::ShapeContainer{h, w});
    py::array_t<int32_t> out_indices(py::array::ShapeContainer{h, w});
    auto out = out_gray.mutable_unchecked<2>();
    auto idx = out_indices.mutable_unchecked<2>();

    for (py::ssize_t y = 0; y < h; ++y) {
        for (py::ssize_t x = 0; x < w; ++x) {
            const int b = static_cast<int>(in(y, x, 0));
            const int g = static_cast<int>(in(y, x, 1));
            const int r = static_cast<int>(in(y, x, 2));
            // OpenCV BGR2GRAY approximation
            const int gray = (114 * b + 587 * g + 299 * r + 500) / 1000;
            const uint8_t quant_idx = quant_lut[static_cast<size_t>(gray)];
            idx(y, x) = static_cast<int32_t>(quant_idx);
            out(y, x) = display_lut[static_cast<size_t>(quant_idx)];
        }
    }

    return py::make_tuple(out_gray, out_indices);
}

py::array_t<double> distribution_from_indices(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
    int levels
) {
    levels = std::max(2, levels);
    auto arr = indices.unchecked<2>();
    std::vector<double> counts(static_cast<size_t>(levels), 0.0);
    const py::ssize_t h = arr.shape(0);
    const py::ssize_t w = arr.shape(1);
    double total = 0.0;

    for (py::ssize_t y = 0; y < h; ++y) {
        for (py::ssize_t x = 0; x < w; ++x) {
            int v = arr(y, x);
            if (v < 0) {
                v = 0;
            } else if (v >= levels) {
                v = levels - 1;
            }
            counts[static_cast<size_t>(v)] += 1.0;
            total += 1.0;
        }
    }

    py::array_t<double> out(py::array::ShapeContainer{levels});
    auto out_mut = out.mutable_unchecked<1>();
    if (total <= 0.0) {
        for (int i = 0; i < levels; ++i) {
            out_mut(i) = 0.0;
        }
        return out;
    }
    for (int i = 0; i < levels; ++i) {
        out_mut(i) = (counts[static_cast<size_t>(i)] / total) * 100.0;
    }
    return out;
}

} // namespace

PYBIND11_MODULE(valuelens_native, m) {
    m.doc() = "Native acceleration for ValueLens quantization pipeline";
    m.def(
        "quantize_fast",
        &quantize_fast,
        py::arg("bgr"),
        py::arg("levels"),
        py::arg("min_value"),
        py::arg("max_value"),
        py::arg("exp_value"),
        py::arg("display_min"),
        py::arg("display_max"),
        py::arg("display_exp")
    );
    m.def(
        "distribution_from_indices",
        &distribution_from_indices,
        py::arg("indices"),
        py::arg("levels")
    );
}

