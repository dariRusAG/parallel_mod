#include "vector_mod.h"
#include "mod_ops.h"
#include "num_threads.h"
#include <thread>
#include <vector>
#include <barrier>

// Функция возведения числа в степень по модулю
IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
    IntegerWord result = 1; // Инициализация результата
    while (power > 0) {
        if (power % 2 != 0) { // Если степень нечётная
            result = mul_mod(result, base, mod); // Умножаем результат на основание по модулю
        }
        power >>= 1; // Делим степень пополам
        base = mul_mod(base, base, mod); // Квадрат основания по модулю
    }
    return result;
}

// Функция для вычисления (-mod) ^ power % mod
IntegerWord word_pow_mod(size_t power, IntegerWord mod) {
    return pow_mod((-mod) % mod, power, mod);
}

// Структура для представления диапазона индексов, обрабатываемых потоком
struct thread_range {
    std::size_t b, e; // Начальный и конечный индексы
};

// Разбиение работы между потоками
thread_range vector_thread_range(size_t n, unsigned T, unsigned t) {
    auto b = n % T; // Остаток от деления общего числа элементов на число потоков
    auto s = n / T; // Базовое количество элементов на поток
    if (t < b) b = ++s * t; // Дополнительный элемент для первых потоков
    else b += s * t;
    size_t e = b + s; // Конечный индекс
    return thread_range(b, e);
}

// Частичные результаты для каждого потока
struct partial_result_t {
    alignas(std::hardware_destructive_interference_size) IntegerWord value;
};

// Функция вычисления остатка от деления многочлена с использованием схемы Горнера
IntegerWord vector_mod(const IntegerWord* V, std::size_t N, IntegerWord mod) {
    size_t T = get_num_threads(); // Получаем количество потоков
    std::vector<std::thread> threads(T - 1); // Создаём потоки, исключая основной
    std::vector<partial_result_t> partial_results(T); // Частичные результаты
    std::barrier<> bar(T); // Барьер синхронизации для потоков

    auto thread_lambda = [V, N, T, mod, &partial_results, &bar](unsigned t) {
        auto [b, e] = vector_thread_range(N, T, t); // Диапазон индексов для текущего потока

        IntegerWord sum = 0;
        // Схема Горнера: вычисляем частичный результат для диапазона [b, e)
        for (std::size_t i = e; b < i;) {
            sum = add_mod(times_word(sum, mod), V[--i], mod); // Постепенное вычисление остатка
        }
        partial_results[t].value = sum; // Сохраняем частичный результат

        // Объединяем результаты между потоками
        for (size_t i = 1, ii = 2; i < T; i = ii, ii += ii) {
            bar.arrive_and_wait(); // Синхронизация
            if (t % ii == 0 && t + i < T) {
                auto neighbor = vector_thread_range(N, T, t + i); // Диапазон соседнего потока
                partial_results[t].value = add_mod(
                    partial_results[t].value,
                    mul_mod(partial_results[t + i].value, word_pow_mod(neighbor.b - b, mod), mod),
                    mod);
            }
        }
        };

    // Запуск потоков
    for (std::size_t i = 1; i < T; ++i) {
        threads[i - 1] = std::thread(thread_lambda, i);
    }
    thread_lambda(0); // Главный поток выполняет лямбду для t=0

    // Ожидание завершения потоков
    for (auto& i : threads) {
        i.join();
    }
    return partial_results[0].value; // Результат объединения всех потоков
}