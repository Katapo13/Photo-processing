#include <iostream>
#include <opencv2/opencv.hpp>
#include "mpi.h"
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // ������������� MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // �������� ���� (�����) �������� ��������
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // �������� ����� ���������� ���������

    // ������� ��������� �����������
    const int newWidth = 4000;   // ������ (�������)
    const int newHeight = 4000;  // ������ (�������)

    // ��������� ������ ����� ����������� ��� ������� ��������
    int blockHeight = newHeight / size;
    // ������ ����� ��� ����������� ����� (3 ������ - RGB)
    int blockSizeKitty = blockHeight * newWidth * 3;
    // ������ ����� ��� ����������� ������ (3 ������ - RGB)
    int blockSizePuppy = blockHeight * newWidth * 3;

    // ��������� ����� ����������� ��� �������� ��������
    vector<uchar> localKittyBlock(blockSizeKitty);
    vector<uchar> localPuppyBlock(blockSizePuppy);

    // ������� ��� �������� ����������� (������ ��� �������� 0)
    Mat resizedKitty, resizedPuppy;

    // �������� ����� ����������
    auto start = chrono::high_resolution_clock::now();

    // ������� ������� (���� 0) ��������� � ������������ ������
    if (rank == 0) {
        // ������ ����������� �� ������
        Mat kitty = imread("D:/cat.jpg");
        Mat puppy = imread("D:/dog.jpg");

        // �������� �� ���������� ������
        if (kitty.empty() || puppy.empty()) {
            cout << "Unable to read image" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);  // ��������� ���������� ���� ���������
        }

        // ��������� ������� ����������� �� 4000x4000
        resize(kitty, resizedKitty, Size(newWidth, newHeight));
        resize(puppy, resizedPuppy, Size(newWidth, newHeight));

        // ������������� ������ ����� ����������
        for (int i = 0; i < size; ++i) {
            // ���������� ������� �������� (ROI) ��� �������� �����
            Rect roi(0, i * blockHeight, newWidth, blockHeight);

            if (i == 0) {
                // ��� �������� 0 �������� ������ ��������
                memcpy(localKittyBlock.data(), resizedKitty(roi).data, blockSizeKitty);
                memcpy(localPuppyBlock.data(), resizedPuppy(roi).data, blockSizePuppy);
            }
            else {
                // ��� ��������� ��������� ���������� ������ ����� MPI
                MPI_Send(resizedKitty(roi).data, blockSizeKitty, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
                MPI_Send(resizedPuppy(roi).data, blockSizePuppy, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // ��������� �������� �������� ���� ����� ������
        MPI_Recv(localKittyBlock.data(), blockSizeKitty, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localPuppyBlock.data(), blockSizePuppy, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ����������� ���������� ������ � ������ OpenCV Mat
    Mat kittyBlock(blockHeight, newWidth, CV_8UC3, localKittyBlock.data());
    Mat puppyBlock(blockHeight, newWidth, CV_8UC3, localPuppyBlock.data());

    // ��������� ����������� (������������ �����)
    Mat kittyGray, puppyDenoised;
    // ������������ ���� ����� � �/�
    cvtColor(kittyBlock, kittyGray, COLOR_BGR2GRAY);
    // ������� ���� � ����� ������
    medianBlur(puppyBlock, puppyDenoised, 5);

    // ���� ����������� �� ������� ��������
    if (rank == 0) {
        // ������� ������� ��� ������ �����������
        Mat fullKittyGray(newHeight, newWidth, CV_8UC1);        // �/� �����������
        Mat fullPuppyDenoised(newHeight, newWidth, CV_8UC3);    // ��������� �� �����

        // �������� ���� �������� 0
        kittyGray.copyTo(fullKittyGray(Rect(0, 0, newWidth, blockHeight)));
        puppyDenoised.copyTo(fullPuppyDenoised(Rect(0, 0, newWidth, blockHeight)));

        // �������� ����� �� ������ ���������
        for (int i = 1; i < size; ++i) {
            // ��������� �� ����� � �������� �����������
            uchar* kittyPtr = fullKittyGray.data + i * blockHeight * newWidth;
            uchar* puppyPtr = fullPuppyDenoised.data + i * blockHeight * newWidth * 3;

            // ����� ������ �� ������ ���������
            MPI_Recv(kittyPtr, blockSizeKitty / 3, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(puppyPtr, blockSizePuppy, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // �������� ����� ����������
        auto finish = chrono::high_resolution_clock::now();
        auto progtime = chrono::duration_cast<chrono::milliseconds>(finish - start);
        cout << "Time: " << progtime.count() << " ms" << endl;

        // ��������� ������������ ����� �������������
        int currentImage = 0;

        // �������� ���� �����������
        while (true) {
            if (currentImage == 0) {  // ����������� �������� �����������
                namedWindow("Kitty", WINDOW_NORMAL);
                resizeWindow("Kitty", 400, 400);
                imshow("Kitty", resizedKitty);

                namedWindow("Puppy", WINDOW_NORMAL);
                resizeWindow("Puppy", 400, 400);
                imshow("Puppy", resizedPuppy);
            }
            else {  // ����������� ������������ �����������
                namedWindow("Kitty", WINDOW_NORMAL);
                resizeWindow("Kitty", 400, 400);
                imshow("Kitty", fullKittyGray);

                namedWindow("Puppy", WINDOW_NORMAL);
                resizeWindow("Puppy", 400, 400);
                imshow("Puppy", fullPuppyDenoised);
            }

            // ��������� ������� ������
            char key = (char)waitKey(0);
            if (key == 'q') {  // �����
                break;
            }
            else if (key == ' ') {  // ������������ ����� �������������
                currentImage = ++currentImage % 2;
            }
        }
    }
    else {
        // �������� ������������ ������ �� ������� �������
        MPI_Send(kittyGray.data, blockSizeKitty / 3, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
        MPI_Send(puppyDenoised.data, blockSizePuppy, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
    }

    // ���������� ������ MPI
    MPI_Finalize();
    return 0;
}