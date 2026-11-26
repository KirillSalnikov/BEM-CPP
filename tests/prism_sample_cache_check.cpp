#include "prism_sample_cache.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    using namespace prism_samples;
    const bool full = argc == 2 && std::string(argv[1]) == "--full";
    if (argc != 1 && !full) return 2;
    char file[] = "/tmp/bem-sample-cache-XXXXXX";
    const int descriptor = mkstemp(file);
    if (descriptor < 0) return 2;
    close(descriptor);
    struct Cleanup {
        const char* path;
        ~Cleanup() { std::remove(path); }
    } cleanup{file};
    const std::string key(64, 'a');
    Vector a(full ? 144*64*183 : 144), b(full ? 144*256*256 : 288), none;
    for (size_t i=0; i<a.size(); ++i) a[i] = {double(i), -double(i)};
    for (size_t i=0; i<b.size(); ++i) b[i] = {double(i+1), 1.25};
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    if (!write(file, key, a, b)) return 3;
    const auto written = Clock::now();
    Vector x(a.size()), y(b.size());
    if (!read(file, key, x, y) || a != x || b != y) return 4;
    const auto loaded = Clock::now();
    if (read(file, std::string(64,'b'), x, y) || read(file, key, x, y, 1)) return 4;
    std::cout << "bytes " << (a.size()+b.size()) * sizeof(Vector::value_type)
              << " write_s " << std::chrono::duration<double>(written-start).count()
              << " verified_read_and_compare_s " << std::chrono::duration<double>(loaded-written).count() << "\n";
    std::fstream modified(file,std::ios::binary|std::ios::in|std::ios::out);
    modified.seekp(-1,std::ios::end);
    modified.put('x');
    modified.close();
    if (read(file, key, x, y) || !write(file, key, a, none, 1) || !read(file, key, x, none, 1)) return 5;
    std::ofstream extra(file, std::ios::binary|std::ios::app);
    extra.put('x');
    extra.close();
    if (read(file, key, x, none, 1) || truncate(file,100) || read(file, key, x, none, 1)) return 6;
    return 0;
}
