#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

    auto start = chrono::high_resolution_clock::now();
    // Чтение изображения из файла
    Mat kitty = imread("cat.jpg");
    Mat puppy = imread("dog.jpg");

    // Проверка на успешность чтения
    if (kitty.empty() || puppy.empty()) {
        cout << "Unable to read image" << endl;
        return -1;
    }

    int newWidth = 4000;  // Ширина (пиксели)
    int newHeight = 4000; // Высота (пиксели)

    // Изменение размера изображений
    Mat resizedKitty, resizedPuppy;
    resize(kitty, resizedKitty, Size(newWidth, newHeight));
    resize(puppy, resizedPuppy, Size(newWidth, newHeight));

    // Инверсия картинки с котёнком
    Mat invertedKitty;
    cvtColor(resizedKitty, invertedKitty, COLOR_BGR2GRAY);
    
    // Удаление шумов в картинке со щенком
    Mat denoisedPuppy;
    int kernelSize = 5; // Размер ядра
    medianBlur(resizedPuppy, denoisedPuppy, kernelSize);

    auto finish = chrono::high_resolution_clock::now();
    auto progtime = chrono::duration_cast<chrono::milliseconds>(finish - start);
    cout << "Time: " << progtime.count() << " ms" << endl;

    // Логика переключения
    int currentImage = 0;  // Переключатель: 0 - исходное, 1 - инверсия/обработка

    while (true) {
        // Отображение соответствующего изображения
        if (currentImage == 0) { //отображение исходных

            namedWindow("Kitty", WINDOW_NORMAL);
            resizeWindow("Kitty", 400, 400);// Размер окна
            imshow("Kitty", resizedKitty);

            namedWindow("Puppy", WINDOW_NORMAL);
            resizeWindow("Puppy", 400, 400);// Размер окна
            imshow("Puppy", resizedPuppy);
        }
        else if (currentImage == 1) {//отображение отредактированных

            //инверсия в чб
            namedWindow("Kitty", WINDOW_NORMAL);
            resizeWindow("Kitty", 400, 400);  // Размер окна
            imshow("Kitty", invertedKitty);
                
            //без шумов
            namedWindow("Puppy", WINDOW_NORMAL);
            resizeWindow("Puppy", 400, 400);  // Размер окна
            imshow("Puppy", denoisedPuppy);
    
        }

        // Обработка нажатий
        char key = (char)waitKey(0);  // Ожидание нажатия клавиши
        if (key == 'q') {
            // Выход из программы (q)
            break;
        }
        else if (key == ' ') {
            // Переключение между изображениями (" ")
            currentImage = ++currentImage % 2;
        }
    }
    
    return 0;
}
