#! /bin/bash

find . -not -path "./3rdparty/*" -regex '.*\.\(hpp\|h\|cuh\|cu\|cpp\)' -exec clang-format -i {} \;
