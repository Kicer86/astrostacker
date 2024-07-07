
module;

#include <filesystem>
#include <ranges>
#include <span>

#include <mlpack/methods/dbscan/dbscan.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>

export module similar_images_picker;


double computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    return cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
}

cv::Mat similarityToDistanceMatrix(const cv::Mat& similarityMatrix)
{
    cv::Mat distanceMatrix = cv::Mat::zeros(similarityMatrix.size(), CV_64F);

    for (int i = 0; i < similarityMatrix.rows; ++i)
        for (int j = 0; j < similarityMatrix.cols; ++j)
            distanceMatrix.at<double>(i, j) = 1.0 - similarityMatrix.at<double>(i, j);

    return distanceMatrix;
}


export std::vector<std::filesystem::path> pickSimilarImages(const std::span<const std::filesystem::path> images, const std::filesystem::path& dir)
{
    const auto grayImagesRange = images | std::views::transform([](const std::filesystem::path& imagePath)
    {
        const cv::Mat image = cv::imread(imagePath);
        cv::Mat gray;
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        return gray;
    });

    const std::vector<cv::Mat> grayImages(grayImagesRange.begin(), grayImagesRange.end());

    const auto imagesCount = images.size();
    cv::Mat similarityMatrix = cv::Mat::zeros(imagesCount, imagesCount, CV_64F);

    //#pragma omp parallel for
    for (size_t i = 0; i < grayImages.size(); ++i)
        for (size_t j = i + 1; j < grayImages.size(); ++j)
        {
            const double ssim = computeSSIM(grayImages[i], grayImages[j]);
            similarityMatrix.at<double>(i, j) = ssim;
            similarityMatrix.at<double>(j, i) = ssim;
        }

    // Convert similarity matrix to distance matrix
    cv::Mat distanceMatrix = similarityToDistanceMatrix(similarityMatrix);

    // Convert distance matrix to Armadillo matrix for mlpack
    arma::mat distances(distanceMatrix.ptr<double>(), distanceMatrix.rows, distanceMatrix.cols, false, true);

    // Run DBSCAN clustering
    arma::Row<size_t> assignments;
    mlpack::DBSCAN<> dbscan(0.2, 3); // Adjust parameters as needed
    dbscan.Cluster(distances, assignments);

    // Find the largest cluster
    std::map<size_t, int> clusterSizes;
    for (size_t i = 0; i < assignments.n_elem; ++i)
        clusterSizes[assignments[i]]++;

    size_t largestCluster = std::max_element(clusterSizes.begin(), clusterSizes.end(),
                                             [](const std::pair<size_t, int>& p1, const std::pair<size_t, int>& p2)
                                             {
                                                 return p1.second < p2.second;
                                             })->first;

    // Extract frames belonging to the largest cluster
    std::vector<std::filesystem::path> goodFrames;
    for (size_t i = 0; i < assignments.n_elem; ++i)
        if (assignments[i] == largestCluster)
            goodFrames.push_back(images[i]);

    std::cout << "Number of good frames: " << goodFrames.size() << std::endl;

    return goodFrames;
}

