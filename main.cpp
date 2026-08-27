#include "Tensor.hpp"
#include <iostream>

using namespace std;

void demoBasica() {
    cout << "creacion de tensores: " << endl;
    Tensor A = Tensor::zeros({2, 3});
    Tensor B = Tensor::ones({2, 3});
    Tensor C = Tensor::random({2, 2}, 0.0, 1.0);
    Tensor D = Tensor::arange(0, 6);
    A.printSize(); B.printSize(); C.printSize(); D.printSize();
    cout << endl;

    cout << "operadores: " << endl;
    Tensor suma = A + B;
    Tensor resta = B - A;
    Tensor prod  = A * B;
    Tensor escalado = B * 2.0;
    suma.print();
    escalado.print();
    cout << endl;

    cout << "View y Unsqeeze:" << endl;
    Tensor E = Tensor::arange(0, 12);
    Tensor F = E.view({3, 4});
    F.print();
    cout << "E tras view (debe quedar vacio, size=" << E.size() << "): ";
    E.print();
    cout << endl;


    Tensor G = Tensor::arange(0, 3);
    Tensor H = G.unsqueeze(0);
    Tensor I = Tensor::arange(0, 3).unsqueeze(1);
    cout << "unsqueeze(0) sobre vector de 3: "; H.printSize();
    cout << "unsqueeze(1) sobre vector de 3: "; I.printSize();
    cout << endl;


    cout << "concat: " << endl;
    Tensor J = Tensor::ones({2, 3});
    Tensor K = Tensor::zeros({2, 3});
    Tensor L = Tensor::concat({J, K}, 0);
    L.printSize();
    cout << endl;

    cout << "dot y matmul: " << endl;
    Tensor v1({3}, {1, 2, 3});
    Tensor v2({3}, {4, 5, 6});
    Tensor d = dot(v1, v2);
    d.print();
    cout << endl;

    Tensor M1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor M2({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor M3 = matmul(M1, M2);
    M3.print();
    cout << endl;

}

void redNeuronal() {
    cout << "RED NEURONAL: " << endl;

    // 1. Tensor de entrada (datos crudos)
    Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    cout << "1.- Entrada: "; input.printSize();

    // 2. view
    Tensor x = input.view({1000, 400});
    cout << "2.- View: "; x.printSize();

    // 3. matmul con pesos W1
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor h1 = matmul(x, W1);
    cout << "3.- Matmul W1: "; h1.printSize();

    // 4. Suma con bias b1 (1 × 100)
    Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
    Tensor h1b = h1 + b1;
    cout << "4.- Suma bias b1: "; h1b.printSize();

    // 5. Activacion ReLU
    Tensor h1r = h1b.relu();
    cout << "5.- ReLU: "; h1r.printSize();

    // 6. matmul con pesos W2
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor h2 = matmul(h1r, W2);
    cout << "6.- Matmul W2: "; h2.printSize();

    // 7. Suma con bias b2 (1 × 10)
    Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
    Tensor h2b = h2 + b2;
    cout << "7.- Suma bias b2: "; h2b.printSize();

    // 8. Activacion Sigmoid
    Tensor output = h2b.sigmoid();
    cout << "8.- Sigmoid: "; output.printSize();
}

int main() {
    demoBasica();
    redNeuronal();
    return 0;
}
