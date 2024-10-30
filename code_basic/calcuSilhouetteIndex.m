function silhouetteIndex = calcuSilhouetteIndex(data, label)
    % Check if the input sizes are valid
    % assert(size(data, 2) == 2, 'Data should be an nx2 matrix.');
    assert(size(label, 1) == size(data, 1), 'Label vector size should match the number of data points.');

    uniqueLabels = unique(label);
    numClusters = numel(uniqueLabels);

    % Initialize arrays to store distances and silhouette values
    pairwiseDistances = pdist2(data, data);
    silhouetteValues = zeros(size(label));

    % Calculate silhouette value for each point
    for i = 1:numel(label)
        clusterIndex = label(i);

        % Calculate average distance to other points in the same cluster (a)
        a = mean(pairwiseDistances(label == clusterIndex, i));

        % Calculate average distance to points in the nearest other cluster (b)
        bValues = zeros(1, numClusters-1);
        otherClusters = setdiff(uniqueLabels, clusterIndex);
        for j = 1:numel(otherClusters)
            otherClusterIndex = otherClusters(j);
            bValues(j) = mean(pairwiseDistances(label == otherClusterIndex, i));
        end
        b = min(bValues);

        % Calculate silhouette value
        silhouetteValues(i) = (b - a) / max(a, b);
    end

    % Calculate overall silhouette index
    silhouetteIndex = mean(silhouetteValues);
end