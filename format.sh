#! /bin/bash

find . -regex '.*\.\(hpp\|h\|cuh\|cu\|cpp\)' -exec clang-format -i {} \;
