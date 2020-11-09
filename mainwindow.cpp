#include <QImage>

#include <opencv2/opencv.hpp>
#include <opencv2/face/facemark.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , _glasses()
    , _mustache()
    , _mouse_nose()
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QImage image;
    image.load(":/images/glasses.jpg");
    image = image.convertToFormat(QImage::Format_RGB888);
    _glasses = cv::Mat(
    image.height(), image.width(), CV_8UC3,
    image.bits(), image.bytesPerLine()).clone();

    image.load(":/images/mustache.jpg");
    image = image.convertToFormat(QImage::Format_RGB888);
    _mustache = cv::Mat(
    image.height(), image.width(), CV_8UC3,
    image.bits(), image.bytesPerLine()).clone();

    image.load(":/images/mouse-nose.jpg");
    image = image.convertToFormat(QImage::Format_RGB888);
    _mouse_nose = cv::Mat(
    image.height(), image.width(), CV_8UC3,
    image.bits(), image.bytesPerLine()).clone();

    cv::CascadeClassifier cc("haarcascade_frontalface_default.xml");
    cv::Ptr<cv::face::Facemark> markDetector = cv::face::createFacemarkLBF();
    markDetector->loadModel("lbfmodel.yaml");

    cv::VideoCapture vc(0);
    if (!vc.isOpened())
        return;

    cv::Mat frame;
    std::vector<cv::Rect> faces;
    cv::Mat grayFrame;
    std::vector<std::vector<cv::Point2f>> shapes;
    while (true) {
        if (cv::waitKey(1) >= 0)
            break;
        vc >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cc.detectMultiScale(grayFrame, faces, 1.3, 5);

        if (markDetector->fit(frame, faces, shapes))
            for (unsigned long i = 0; i < faces.size(); ++i) {
                drawGlasses(frame, shapes[i]);
                drawMouseNose(frame, shapes[i]);
                drawMustache(frame, shapes[i]);
            }

        cv::imshow("Video", frame);
    }
    vc.release();
    cv::destroyAllWindows();
}

void MainWindow::drawGlasses(cv::Mat &frame, std::vector<cv::Point2f> &marks)
{
    // resize
    cv::Mat ornament;
    double distance = cv::norm(marks[45] - marks[36]) * 1.5;
    cv::resize(_glasses, ornament, cv::Size(0, 0), distance / _glasses.cols, distance / _glasses.cols
               , cv::INTER_NEAREST);

    // rotate
    double angle = -atan((marks[45].y - marks[36].y) / (marks[45].x - marks[36].x));
    cv::Point2f center = cv::Point(ornament.cols/2, ornament.rows/2);
    cv::Mat rotateMatrix = cv::getRotationMatrix2D(center, angle * 180 / 3.14, 1.0);

    cv::Mat rotated;
    cv::warpAffine(
    ornament, rotated, rotateMatrix, ornament.size(),
    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // paint
    center = cv::Point((marks[45].x + marks[36].x) / 2, (marks[45].y + marks[36].y) / 2);
    cv::Rect rec(center.x - rotated.cols / 2, center.y - rotated.rows / 2, rotated.cols, rotated.rows);
    frame(rec) &= rotated;
}

void MainWindow::drawMustache(cv::Mat &frame, std::vector<cv::Point2f> &marks)
{
    // resize
    cv::Mat ornament;
    double distance = cv::norm(marks[54] - marks[48]) * 1.5;
    cv::resize(_mustache, ornament, cv::Size(0, 0), distance / _mustache.cols, distance / _mustache.cols
               , cv::INTER_NEAREST);

    // rotate
    double angle = -atan((marks[54].y - marks[48].y) / (marks[54].x - marks[48].x));
    cv::Point2f center = cv::Point(ornament.cols/2, ornament.rows/2);
    cv::Mat rotateMatrix = cv::getRotationMatrix2D(center, angle * 180 / 3.14, 1.0);

    cv::Mat rotated;
    cv::warpAffine(
        ornament, rotated, rotateMatrix, ornament.size(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // paint
    center = cv::Point((marks[33].x + marks[51].x) / 2, (marks[33].y + marks[51].y) / 2);
    cv::Rect rec(center.x - rotated.cols / 2, center.y - rotated.rows / 2, rotated.cols, rotated.rows);
    frame(rec) &= rotated;
}

void MainWindow::drawMouseNose(cv::Mat &frame, std::vector<cv::Point2f> &marks)
{
    // resize
    cv::Mat ornament;
    double distance = cv::norm(marks[13] - marks[3]);
    cv::resize(_mouse_nose, ornament, cv::Size(0, 0), distance / _mouse_nose.cols, distance / _mouse_nose.cols
               , cv::INTER_NEAREST);

    // rotate
    double angle = -atan((marks[16].y - marks[0].y) / (marks[16].x - marks[0].x));
    cv::Point2f center = cv::Point(ornament.cols/2, ornament.rows/2);
    cv::Mat rotateMatrix = cv::getRotationMatrix2D(center, angle * 180 / 3.14, 1.0);

    cv::Mat rotated;
    cv::warpAffine(
        ornament, rotated, rotateMatrix, ornament.size(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // paint
    center = marks[30];
    cv::Rect rec(center.x - rotated.cols / 2, center.y - rotated.rows / 2, rotated.cols, rotated.rows);
    frame(rec) &= rotated;
}
