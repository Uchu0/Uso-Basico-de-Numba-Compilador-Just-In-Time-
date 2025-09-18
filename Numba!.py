import numpy as np
from numba import njit
import time


# 1. Función normal (sin optimizar)
def operaciones_matrices_normal(A, B):
    """Suma y multiplicación elemento por elemento"""
    suma = A + B
    multiplicacion = A * B
    return suma, multiplicacion


# 2. Función optimizada con Numba
@njit
def operaciones_matrices_numba(A, B):
    """Suma y multiplicación elemento por elemento - optimizado"""
    suma = A + B
    multiplicacion = A * B
    return suma, multiplicacion


# 3. Función para crear matriz de ejemplo
def crear_matriz_ejemplo(filas, columnas):
    """Crea una matriz con valores secuenciales"""
    return np.arange(filas * columnas, dtype=np.float64).reshape(filas, columnas)


# 4. Función principal
def main():
    print("🎯 EJERCICIO SIMPLE CON NUMBA 🎯")
    print("=" * 40)

    # Crear matrices pequeñas para ejemplo visual
    A = crear_matriz_ejemplo(2, 3)
    B = crear_matriz_ejemplo(2, 3) + 1  # B = A + 1

    print("Matriz A:")
    print(A)
    print("\nMatriz B:")
    print(B)

    # Operaciones sin Numba
    print("\n🔹 SIN NUMBA:")
    suma_normal, mult_normal = operaciones_matrices_normal(A, B)
    print("Suma A + B:")
    print(suma_normal)
    print("\nMultiplicación elemento por elemento A * B:")
    print(mult_normal)

    # Operaciones con Numba
    print("\n🔹 CON NUMBA:")
    suma_numba, mult_numba = operaciones_matrices_numba(A, B)
    print("Suma A + B:")
    print(suma_numba)
    print("\nMultiplicación elemento por elemento A * B:")
    print(mult_numba)

    # Verificar que los resultados son iguales
    print("\n✅ Verificación:")
    print(f"Sumas iguales: {np.array_equal(suma_normal, suma_numba)}")
    print(f"Multiplicaciones iguales: {np.array_equal(mult_normal, mult_numba)}")

    # Comparación de rendimiento con matrices más grandes
    print("\n" + "=" * 40)
    print("⚡ COMPARACIÓN DE RENDIMIENTO:")

    # Crear matrices grandes
    tamaño = 1000
    A_grande = np.random.rand(tamaño, tamaño)
    B_grande = np.random.rand(tamaño, tamaño)

    print(f"Matrices de {tamaño}x{tamaño} elementos")

    # Medir tiempo sin Numba
    inicio = time.time()
    operaciones_matrices_normal(A_grande, B_grande)
    tiempo_normal = time.time() - inicio

    # Medir tiempo con Numba (primera vez - incluye compilación)
    inicio = time.time()
    operaciones_matrices_numba(A_grande, B_grande)
    tiempo_numba_1 = time.time() - inicio

    # Medir tiempo con Numba (segunda vez - solo ejecución)
    inicio = time.time()
    operaciones_matrices_numba(A_grande, B_grande)
    tiempo_numba_2 = time.time() - inicio

    print(f"\nTiempo sin Numba: {tiempo_normal:.4f} segundos")
    print(f"Tiempo con Numba (1ra vez): {tiempo_numba_1:.4f} segundos")
    print(f"Tiempo con Numba (2da vez): {tiempo_numba_2:.4f} segundos")
    print(f"Aceleración: {tiempo_normal / tiempo_numba_2:.1f}x más rápido!")

    # Ejemplo adicional: Suma de todos los elementos
    print("\n" + "=" * 40)
    print("➕ SUMA DE TODOS LOS ELEMENTOS:")

    @njit
    def suma_total_numba(matriz):
        total = 0.0
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                total += matriz[i, j]
        return total

    # Comparar con NumPy
    matriz = np.random.rand(500, 500)

    inicio = time.time()
    suma_numpy = np.sum(matriz)
    tiempo_numpy = time.time() - inicio

    inicio = time.time()
    suma_numba = suma_total_numba(matriz)
    tiempo_numba = time.time() - inicio

    print(f"Suma con NumPy: {suma_numpy:.6f} ({tiempo_numpy:.6f} segundos)")
    print(f"Suma con Numba: {suma_numba:.6f} ({tiempo_numba:.6f} segundos)")
    print(f"Diferencia: {abs(suma_numpy - suma_numba):.10e}")


if __name__ == "__main__":
    main()