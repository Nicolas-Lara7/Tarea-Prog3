#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>
#include <string>

using namespace std;



class Tensor {

    double* data_;
    vector<size_t> shape_;
    size_t totalSize_;

    static size_t tamanio(const vector<size_t>& shape);
    size_t indice(const vector<size_t>& idx) const;
    void LiberarEspacio();

    // Broadcasting estilo NumPy (necesario, p.ej., para sumar un bias 1x100
    // a un batch 1000x100, como exige la red neuronal del enunciado).
    static vector<size_t> broadcastShape(const vector<size_t>& a, const vector<size_t>& b);
    static size_t broadcastindice(const vector<size_t>& idx, const vector<size_t>& shape);
    template <typename Op>
    Tensor elementwiseOp(const Tensor& other, Op op) const;

public:
    //constructores
    Tensor();                                                  // constructor vaico
    Tensor(const vector<size_t>& shape,
           const vector<double>& values);

    Tensor(const Tensor& other);                    // copia profunda
    Tensor(Tensor&& other) noexcept;                // movimiento
    Tensor& operator=(const Tensor& other);         // asignación por copia
    Tensor& operator=(Tensor&& other) noexcept;     // asignación por movimiento
    ~Tensor();                                      // destructor

    //tensores pre-definidos
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape, double min, double max);
    static Tensor arange(double start, double end);

    //sobrecarga de operadores
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    //modificar dimensiones
    Tensor view(const vector<size_t>& newShape);         // no copia: mueve los datos
    Tensor unsqueeze(size_t dim);                        // no copia: mueve los datos

    //concatenar
    static Tensor concat(const vector<Tensor>& tensors, size_t dim);

    //funciones amigas
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    //activaciones
    Tensor relu() const;
    Tensor sigmoid() const;

    //extras
    const vector<size_t>& shape() const { return shape_; }
    size_t size() const { return totalSize_; }
    double& at(const vector<size_t>& idx);
    const double& at(const vector<size_t>& idx) const;
    void print() const;
    void printSize() const;


};

#endif // TENSOR_HPP
