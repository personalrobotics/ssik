// THE GATE (data-driven): every self-contained artifact validated against its
// Python oracle golden, zero pybind/Python. The arm list is generated into
// artifact_gate.hpp from cpp/gen by scripts/cpp_emit.py, so adding an arm needs
// no edit here -- re-emit and it auto-enrols. Run `cpp_emit.py --all` first to
// (re)generate the gate + goldens.
#include "artifact_gate.hpp"

int main() { return ssik::artifact_test::run_all(); }
