#include "precompiled_stl.hpp"

using namespace std;

#include "utils.hpp"
#include "core_functions.hpp"
#include "image_functions.hpp"
#include "visu.hpp"
#include "read.hpp"
#include <string>
int main(int argc, char**argv) {
  string bpath = "";
  if (argc >= 2) {
      bpath = argv[1];
      cout << "Using bpath: " << bpath << endl;
  } else {
      bpath = "";
      cout << "No bpath provided. Using default." << endl;
  }

  vector<Sample> sample = readAll("test", -1, bpath);
  cout << sample.size() << endl;
}
