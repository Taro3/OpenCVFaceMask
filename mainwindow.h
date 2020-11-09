#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat _glasses;
    cv::Mat _mustache;
    cv::Mat _mouse_nose;

    void drawGlasses(cv::Mat &frame, std::vector<cv::Point2f>&marks);
    void drawMustache(cv::Mat &frame, std::vector<cv::Point2f> &marks);
    void drawMouseNose(cv::Mat &frame, std::vector<cv::Point2f> &marks);
};
#endif // MAINWINDOW_H
