
module;

#include <filesystem>
#include <ranges>
#include <span>

#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>

export module similar_images_picker;


double computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    return cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
}

struct Edge {
    int u, v;
    double weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

int findParent(size_t i, std::vector<int>& parent) {
    if (parent[i] == i)
        return i;
    return parent[i] = findParent(parent[i], parent);
}

void unionSets(int i, int j, std::vector<int>& parent, std::vector<int>& rank) {
    int root1 = findParent(i, parent);
    int root2 = findParent(j, parent);
    if (root1 != root2) {
        if (rank[root1] > rank[root2])
            parent[root2] = root1;
        else if (rank[root1] < rank[root2])
            parent[root1] = root2;
        else {
            parent[root2] = root1;
            rank[root1]++;
        }
    }
}


export void pickSimilarImages(const std::span<const std::filesystem::path> images, const std::filesystem::path& dir)
{
    const auto grayImagesRange = images | std::views::transform([](const std::filesystem::path& imagePath)
    {
        const cv::Mat image = cv::imread(imagePath);
        cv::Mat gray;
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        return gray;
    });

    const std::vector<cv::Mat> grayImages(grayImagesRange.begin(), grayImagesRange.end());

    std::vector<std::vector<double>> ssimScores(images.size(), std::vector<double>(images.size(), 0));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < grayImages.size(); ++i) {
        for (size_t j = i + 1; j < grayImages.size(); ++j) {
            double ssim = computeSSIM(grayImages[i], grayImages[j]);
            ssimScores[i][j] = ssim;
            ssimScores[j][i] = ssim;
        }
    }

    std::priority_queue<Edge> edges;
    for (size_t i = 0; i < ssimScores.size(); ++i) {
        for (size_t j = i + 1; j < ssimScores[i].size(); ++j) {
            edges.push({static_cast<int>(i), static_cast<int>(j), ssimScores[i][j]});
        }
    }

    // Initialize union-find structures
    std::vector<int> parent(images.size()), rank(images.size(), 0);
    for (size_t i = 0; i < images.size(); ++i) {
        parent[i] = i;
    }

    // Agglomerative clustering
    double threshold = 0.9; // Adjust this threshold based on your requirement
    while (!edges.empty() && edges.top().weight > threshold)
    {
        Edge e = edges.top();
        edges.pop();
        unionSets(e.u, e.v, parent, rank);
    }

    // Group images by their root parent
    std::map<int, std::vector<int>> clusters;
    for (size_t i = 0; i < parent.size(); ++i)
    {
        int root = findParent(i, parent);
        clusters[root].push_back(i);
    }

    // Find the largest cluster
    std::vector<int> largestCluster;
    for (const auto& cluster : clusters)
    {
        if (cluster.second.size() > largestCluster.size()) {
            largestCluster = cluster.second;
        }
    }

    // Output the largest cluster
    std::cout << "Largest cluster size: " << largestCluster.size() << std::endl;
    for (int index : largestCluster)
    {
        std::cout << "Image: " << images[index] << std::endl;
    }
}

