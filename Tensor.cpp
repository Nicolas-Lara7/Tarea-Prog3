#include "Tensor.hpp"
#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>
using namespace std;






size_t Tensor::tamanio(const vector<size_t>& shape) {
    if (shape.empty() || shape.size() > 3)
        throw invalid_argument("La forma del tensor debe tener entre 1 y 3 dimensiones.");
    size_t total = 1;
    for (size_t d : shape) {
        if (d == 0)
            throw invalid_argument("Las dimensiones del tensor no pueden ser 0.");
        total *= d;
    }
    return total;
}

size_t Tensor::indice(const vector<size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw invalid_argument("numero de indices no coincide con las dimensiones.");
    size_t flat = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (idx[i] >= shape_[i])
            throw out_of_range("indice fuera de rango.");
        flat += idx[i] * stride;
        stride *= shape_[i];
    }
    return flat;
}

void Tensor::LiberarEspacio() {
    delete[] data_;
    data_ = nullptr;
}

Tensor::Tensor() : data_(nullptr), shape_(), totalSize_(0) {}

Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values): data_(nullptr), shape_(shape), totalSize_(0) {
    totalSize_ = tamanio(shape_);
    if (values.size() != totalSize_)
        throw invalid_argument("El tamano de 'values' no coincide con el producto de las dimensiones.");
    data_ = new double[totalSize_];
    copy(values.begin(), values.end(), data_);
}

Tensor::Tensor(const Tensor& other): data_(nullptr), shape_(other.shape_), totalSize_(other.totalSize_) {
    if (totalSize_ > 0) {
        data_ = new double[totalSize_];
        copy(other.data_, other.data_ + totalSize_, data_);
    }
}


Tensor::Tensor(Tensor&& other) noexcept: data_(other.data_), shape_(move(other.shape_)), totalSize_(other.totalSize_) {
    other.data_ = nullptr;
    other.totalSize_ = 0;
    other.shape_.clear();
}


Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    LiberarEspacio();
    shape_ = other.shape_;
    totalSize_ = other.totalSize_;
    if (totalSize_ > 0) {
        data_ = new double[totalSize_];
        copy(other.data_, other.data_ + totalSize_, data_);
    } else {
        data_ = nullptr;
    }
    return *this;
}


Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    LiberarEspacio();
    data_ = other.data_;
    shape_ = move(other.shape_);
    totalSize_ = other.totalSize_;
    other.data_ = nullptr;
    other.totalSize_ = 0;
    other.shape_.clear();
    return *this;
}

Tensor::~Tensor() {
    LiberarEspacio();
}



Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total = tamanio(shape);
    return Tensor(shape, vector<double>(total, 0.0));
}

Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t total = tamanio(shape);
    return Tensor(shape, vector<double>(total, 1.0));
}

Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    size_t total = tamanio(shape);
    vector<double> values(total);
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(min, max);
    for (auto& v : values) v = dist(gen);
    return Tensor(shape, values);
}

Tensor Tensor::arange(double start, double end) {
    if (end <= start)
        throw invalid_argument(" 'End' debe ser mayor que 'start'.");
    vector<double> values;
    for (double v = start; v < end; v += 1.0) values.push_back(v);
    return Tensor({values.size()}, values);
}


vector<size_t> Tensor::broadcastShape(const vector<size_t>& a, const vector<size_t>& b) {
    size_t n = max(a.size(), b.size());
    vector<size_t> pa(n, 1), pb(n, 1), result(n, 1);
    copy(a.rbegin(), a.rend(), pa.rbegin());
    copy(b.rbegin(), b.rend(), pb.rbegin());
    for (size_t i = 0; i < n; ++i) {
        if (pa[i] == pb[i] || pa[i] == 1 || pb[i] == 1)
            result[i] = max(pa[i], pb[i]);
        else
            throw invalid_argument("Dimensiones incompatibles para la operacion.");
    }
    if (result.size() > 3)
        throw invalid_argument("El resultado no puede exceder 3 dimensiones.");
    return result;
}

size_t Tensor::broadcastindice(const vector<size_t>& idx, const vector<size_t>& shape) {
    size_t n = idx.size();
    vector<size_t> padded(n, 1);
    copy(shape.rbegin(), shape.rend(), padded.rbegin());

    size_t flat = 0, stride = 1;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        size_t coord = (padded[i] == 1) ? 0 : idx[i];
        flat += coord * stride;
        stride *= padded[i];
    }
    return flat;
}

template <typename Op>
Tensor Tensor::elementwiseOp(const Tensor& other, Op op) const {
    vector<size_t> outShape = broadcastShape(shape_, other.shape_);
    size_t total = tamanio(outShape);
    vector<double> result(total);


    vector<size_t> idx(outShape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear) {
        size_t rem = linear;
        for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
            idx[d] = rem % outShape[d];
            rem /= outShape[d];
        }
        size_t ia = broadcastindice(idx, shape_);
        size_t ib = broadcastindice(idx, other.shape_);
        result[linear] = op(data_[ia], other.data_[ib]);
    }
    return Tensor(outShape, result);
}


//OPERADORES
Tensor Tensor::operator+(const Tensor& other) const {
    return elementwiseOp(other, [](double a, double b) { return a + b; });
}

Tensor Tensor::operator-(const Tensor& other) const {
    return elementwiseOp(other, [](double a, double b) { return a - b; });
}

Tensor Tensor::operator*(const Tensor& other) const {
    return elementwiseOp(other, [](double a, double b) { return a * b; });
}

Tensor Tensor::operator*(double scalar) const {
    vector<double> result(totalSize_);
    for (size_t i = 0; i < totalSize_; ++i) result[i] = data_[i] * scalar;
    return Tensor(shape_, result);
}

//VIEW Y UNSQUEEZE

Tensor Tensor::view(const vector<size_t>& newShape) {
    size_t newTotal = tamanio(newShape);
    if (newTotal != totalSize_)
        throw invalid_argument("El numero total de elementos debe mantenerse.");

    Tensor result;
    result.data_ = data_;
    result.shape_ = newShape;
    result.totalSize_ = totalSize_;


    data_ = nullptr;
    shape_.clear();
    totalSize_ = 0;
    return result;
}

Tensor Tensor::unsqueeze(size_t dim) {
    if (shape_.size() >= 3)
        throw invalid_argument("El resultado excederia 3 dimensiones en unsqueeze.");
    if (dim > shape_.size())
        throw out_of_range("La posicion es invalida.");

    vector<size_t> newShape = shape_;
    newShape.insert(newShape.begin() + dim, 1);

    Tensor result;
    result.data_ = data_;
    result.shape_ = newShape;
    result.totalSize_ = totalSize_;

    data_ = nullptr;
    shape_.clear();
    totalSize_ = 0;
    return result;
}



Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty())
        throw invalid_argument("La lista de tensores esta vacia.");

    const vector<size_t>& baseShape = tensors[0].shape_;
    if (dim >= baseShape.size())
        throw invalid_argument("La dimension es invalida.");

    vector<size_t> outShape = baseShape;
    size_t concatDimTotal = 0;
    for (const auto& t : tensors) {
        if (t.shape_.size() != baseShape.size())
            throw invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones.");
        for (size_t i = 0; i < baseShape.size(); ++i) {
            if (i == dim) continue;
            if (t.shape_[i] != baseShape[i])
                throw invalid_argument("Las dimensiones son incompatibles fuera del eje de concatenacion.");
        }
        concatDimTotal += t.shape_[dim];
    }
    outShape[dim] = concatDimTotal;

    size_t total = tamanio(outShape);
    vector<double> result(total);

    vector<size_t> idx(outShape.size(), 0);
    size_t offsetAlongDim = 0;
    for (const auto& t : tensors) {
        for (size_t linear = 0; linear < t.totalSize_; ++linear) {
            size_t rem = linear;
            for (int d = static_cast<int>(t.shape_.size()) - 1; d >= 0; --d) {
                idx[d] = rem % t.shape_[d];
                rem /= t.shape_[d];
            }
            vector<size_t> outIdx = idx;
            outIdx[dim] += offsetAlongDim;

            size_t outFlat = 0, stride = 1;
            for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
                outFlat += outIdx[d] * stride;
                stride *= outShape[d];
            }
            result[outFlat] = t.data_[linear];
        }
        offsetAlongDim += t.shape_[dim];
    }

    return Tensor(outShape, result);
}

//FUNCIONES AMIGAS

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape_ != b.shape_)
        throw invalid_argument("Los tensores deben tener la misma forma.");
    double acc = 0.0;
    for (size_t i = 0; i < a.totalSize_;
        ++i) acc += a.data_[i] * b.data_[i];
    Tensor result({1}, {acc});
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2)
        throw invalid_argument("Los tensores deben ser 2D.");
    size_t n = a.shape_[0], k = a.shape_[1];
    size_t k2 = b.shape_[0], m = b.shape_[1];
    if (k != k2)
        throw invalid_argument("Las dimensiones internas son incompatibles.");

    vector<double> result(n * m, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t p = 0; p < k; ++p) {
            double aip = a.data_[i * k + p];
            if (aip == 0.0) continue;
            for (size_t j = 0; j < m; ++j) {
                result[i * m + j] += aip * b.data_[p * m + j];
            }
        }
    }
    Tensor out({n, m}, result);
    return out;
}

//ACTIVACIONES

Tensor Tensor::relu() const {
    vector<double> result(totalSize_);
    for (size_t i = 0; i < totalSize_;
        ++i) result[i] = max(0.0, data_[i]);
    return Tensor(shape_, result);
}

Tensor Tensor::sigmoid() const {
    vector<double> result(totalSize_);
    for (size_t i = 0; i < totalSize_;
        ++i) result[i] = 1.0 / (1.0 + exp(-data_[i]));
    return Tensor(shape_, result);
}

//EXTRAS

double& Tensor::at(const vector<size_t>& idx) {
    return data_[indice(idx)];
}

const double& Tensor::at(const vector<size_t>& idx) const {
    return data_[indice(idx)];
}

void Tensor::print() const {
    cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    cout << "], data=[";
    for (size_t i = 0; i < totalSize_; ++i) {
        cout << data_[i] << (i + 1 < totalSize_ ? ", " : "");
    }
    cout << "])" << endl;
}

void Tensor::printSize() const {
    cout << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    cout << "]" << endl;
}
