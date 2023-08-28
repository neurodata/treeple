#include <vector>
#include <cstddef>

struct VectorHash {
    /**
     * @brief A hash function for the contents of a vector of ints.
     * 
     * Assumes that the vector is sorted.
     * 
     * @param v vector
     * @return size_t the key.
     */
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = v.size();
        for (const int& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
