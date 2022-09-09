#!/bin/sh
wget https://www.eth3d.net/data/observatory_dslr_undistorted.7z
wget https://www.eth3d.net/data/living_room_dslr_undistorted.7z

7z x observatory_dslr_undistorted.7z -o.
7z x living_room_dslr_undistorted.7z -o.
