#include <iostream>
#include <opencv2/opencv.hpp>
#include "mpi.h"
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Получаем ранг (номер) текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Получаем общее количество процессов

    // Размеры итогового изображения
    const int newWidth = 4000;   // Ширина (пиксели)
    const int newHeight = 4000;  // Высота (пиксели)

    // Вычисляем высоту блока изображения для каждого процесса
    int blockHeight = newHeight / size;
    // Размер блока для изображения кошки (3 канала - RGB)
    int blockSizeKitty = blockHeight * newWidth * 3;
    // Размер блока для изображения собаки (3 канала - RGB)
    int blockSizePuppy = blockHeight * newWidth * 3;

    // Локальные блоки изображений для текущего процесса
    vector<uchar> localKittyBlock(blockSizeKitty);
    vector<uchar> localPuppyBlock(blockSizePuppy);

    // Матрицы для хранения изображений (только для процесса 0)
    Mat resizedKitty, resizedPuppy;

    // Засекаем время выполнения
    auto start = chrono::high_resolution_clock::now();

    // Главный процесс (ранг 0) загружает и распределяет данные
    if (rank == 0) {
        // Чтение изображений из файлов
        Mat kitty = imread("D:/cat.jpg");
        Mat puppy = imread("D:/dog.jpg");

        // Проверка на успешность чтения
        if (kitty.empty() || puppy.empty()) {
            cout << "Unable to read image" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);  // Аварийное завершение всех процессов
        }

        // Изменение размера изображений до 4000x4000
        resize(kitty, resizedKitty, Size(newWidth, newHeight));
        resize(puppy, resizedPuppy, Size(newWidth, newHeight));

        // Распределение данных между процессами
        for (int i = 0; i < size; ++i) {
            // Определяем область интереса (ROI) для текущего блока
            Rect roi(0, i * blockHeight, newWidth, blockHeight);

            if (i == 0) {
                // Для процесса 0 копируем данные напрямую
                memcpy(localKittyBlock.data(), resizedKitty(roi).data, blockSizeKitty);
                memcpy(localPuppyBlock.data(), resizedPuppy(roi).data, blockSizePuppy);
            }
            else {
                // Для остальных процессов отправляем данные через MPI
                MPI_Send(resizedKitty(roi).data, blockSizeKitty, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
                MPI_Send(resizedPuppy(roi).data, blockSizePuppy, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // Остальные процессы получают свои блоки данных
        MPI_Recv(localKittyBlock.data(), blockSizeKitty, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localPuppyBlock.data(), blockSizePuppy, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Преобразуем полученные данные в формат OpenCV Mat
    Mat kittyBlock(blockHeight, newWidth, CV_8UC3, localKittyBlock.data());
    Mat puppyBlock(blockHeight, newWidth, CV_8UC3, localPuppyBlock.data());

    // Обработка изображений (параллельная часть)
    Mat kittyGray, puppyDenoised;
    // Конвертируем блок кошки в ч/б
    cvtColor(kittyBlock, kittyGray, COLOR_BGR2GRAY);
    // Убираем шумы с блока собаки
    medianBlur(puppyBlock, puppyDenoised, 5);

    // Сбор результатов на главном процессе
    if (rank == 0) {
        // Создаем матрицы для полных изображений
        Mat fullKittyGray(newHeight, newWidth, CV_8UC1);        // Ч/Б изображение
        Mat fullPuppyDenoised(newHeight, newWidth, CV_8UC3);    // Очищенное от шумов

        // Копируем блок процесса 0
        kittyGray.copyTo(fullKittyGray(Rect(0, 0, newWidth, blockHeight)));
        puppyDenoised.copyTo(fullPuppyDenoised(Rect(0, 0, newWidth, blockHeight)));

        // Получаем блоки от других процессов
        for (int i = 1; i < size; ++i) {
            // Указатели на место в итоговом изображении
            uchar* kittyPtr = fullKittyGray.data + i * blockHeight * newWidth;
            uchar* puppyPtr = fullPuppyDenoised.data + i * blockHeight * newWidth * 3;

            // Прием данных от других процессов
            MPI_Recv(kittyPtr, blockSizeKitty / 3, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(puppyPtr, blockSizePuppy, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Замеряем время выполнения
        auto finish = chrono::high_resolution_clock::now();
        auto progtime = chrono::duration_cast<chrono::milliseconds>(finish - start);
        cout << "Time: " << progtime.count() << " ms" << endl;

        // Интерфейс переключения между изображениями
        int currentImage = 0;

        // Основной цикл отображения
        while (true) {
            if (currentImage == 0) {  // Отображение исходных изображений
                namedWindow("Kitty", WINDOW_NORMAL);
                resizeWindow("Kitty", 400, 400);
                imshow("Kitty", resizedKitty);

                namedWindow("Puppy", WINDOW_NORMAL);
                resizeWindow("Puppy", 400, 400);
                imshow("Puppy", resizedPuppy);
            }
            else {  // Отображение обработанных изображений
                namedWindow("Kitty", WINDOW_NORMAL);
                resizeWindow("Kitty", 400, 400);
                imshow("Kitty", fullKittyGray);

                namedWindow("Puppy", WINDOW_NORMAL);
                resizeWindow("Puppy", 400, 400);
                imshow("Puppy", fullPuppyDenoised);
            }

            // Обработка нажатий клавиш
            char key = (char)waitKey(0);
            if (key == 'q') {  // Выход
                break;
            }
            else if (key == ' ') {  // Переключение между изображениями
                currentImage = ++currentImage % 2;
            }
        }
    }
    else {
        // Отправка обработанных данных на главный процесс
        MPI_Send(kittyGray.data, blockSizeKitty / 3, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
        MPI_Send(puppyDenoised.data, blockSizePuppy, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}