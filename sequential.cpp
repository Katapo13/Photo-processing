#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

    auto start = chrono::high_resolution_clock::now();
    // ������ ����������� �� �����
    Mat kitty = imread("cat.jpg");
    Mat puppy = imread("dog.jpg");

    // �������� �� ���������� ������
    if (kitty.empty() || puppy.empty()) {
        cout << "Unable to read image" << endl;
        return -1;
    }

    int newWidth = 4000;  // ������ (�������)
    int newHeight = 4000; // ������ (�������)

    // ��������� ������� �����������
    Mat resizedKitty, resizedPuppy;
    resize(kitty, resizedKitty, Size(newWidth, newHeight));
    resize(puppy, resizedPuppy, Size(newWidth, newHeight));

    // �������� �������� � �������
    Mat invertedKitty;
    cvtColor(resizedKitty, invertedKitty, COLOR_BGR2GRAY);
    
    // �������� ����� � �������� �� ������
    Mat denoisedPuppy;
    int kernelSize = 5; // ������ ����
    medianBlur(resizedPuppy, denoisedPuppy, kernelSize);

    auto finish = chrono::high_resolution_clock::now();
    auto progtime = chrono::duration_cast<chrono::milliseconds>(finish - start);
    cout << "Time: " << progtime.count() << " ms" << endl;

    // ������ ������������
    int currentImage = 0;  // �������������: 0 - ��������, 1 - ��������/���������

    while (true) {
        // ����������� ���������������� �����������
        if (currentImage == 0) { //����������� ��������

            namedWindow("Kitty", WINDOW_NORMAL);
            resizeWindow("Kitty", 400, 400);// ������ ����
            imshow("Kitty", resizedKitty);

            namedWindow("Puppy", WINDOW_NORMAL);
            resizeWindow("Puppy", 400, 400);// ������ ����
            imshow("Puppy", resizedPuppy);
        }
        else if (currentImage == 1) {//����������� �����������������

            //�������� � ��
            namedWindow("Kitty", WINDOW_NORMAL);
            resizeWindow("Kitty", 400, 400);  // ������ ����
            imshow("Kitty", invertedKitty);
                
            //��� �����
            namedWindow("Puppy", WINDOW_NORMAL);
            resizeWindow("Puppy", 400, 400);  // ������ ����
            imshow("Puppy", denoisedPuppy);
    
        }

        // ��������� �������
        char key = (char)waitKey(0);  // �������� ������� �������
        if (key == 'q') {
            // ����� �� ��������� (q)
            break;
        }
        else if (key == ' ') {
            // ������������ ����� ������������� (" ")
            currentImage = ++currentImage % 2;
        }
    }
    
    return 0;
}
