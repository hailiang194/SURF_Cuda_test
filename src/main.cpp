#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <ctime>
#include <iostream>

int main()
{
    cv::Mat image = cv::imread("../image.png", cv::IMREAD_COLOR);
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    //CPU
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    std::clock_t start = std::clock();
    surf->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors); 
    std::cout << "CPU Time: " << (std::clock() - start) << std::endl;

    cv::Mat cpuKeyPointsImage;
    cv::drawKeypoints(grayImage, keypoints, cpuKeyPointsImage);

    //GPU
    std::cout << "Enable device: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    cv::cuda::GpuMat gpuGrayImage;
    gpuGrayImage.upload(grayImage);
    
    cv::Ptr<cv::cuda::SURF_CUDA> surfCuda = cv::cuda::SURF_CUDA::create(100);
    
    cv::cuda::GpuMat gpuKeypoints;
    cv::cuda::GpuMat gpuDescriptors;
    cv::cuda::GpuMat gpuNoArray;
    gpuNoArray.upload(cv::noArray());
    
    start = std::clock();
    surfCuda->detectWithDescriptors(gpuGrayImage, gpuNoArray, gpuKeypoints, gpuDescriptors);
    std::cout << "GPU Time: " << (std::clock() - start) << std::endl;

    std::vector<cv::KeyPoint> gpuConvertedKeypoints;
    surfCuda->downloadKeypoints(gpuKeypoints, gpuConvertedKeypoints);

    cv::Mat gpuKeyPointsImage;
    cv::drawKeypoints(grayImage, gpuConvertedKeypoints, gpuKeyPointsImage);

    cv::imshow("CPU", cpuKeyPointsImage);
    cv::imshow("GPU", gpuKeyPointsImage);
    cv::waitKey(0);
    return 0;
}
