#!/usr/bin/env bash


binary_modules=(
gas
perl
python
ffmpeg
libMagickCore-7.Q16HDRI.so.6.0.0
libopencv_core.so.4.0.1
libxerces-c-3.2.so
)


export LD_LIBRARY_PATH="$PWD/local/lib;$PWD/local/usr/lib"
bcov_bin_path="$PWD/local/bin/bcov"
bcov_rt_lib_path="$PWD/local/lib/libbcov-rt.so"


# Patching all sample binaries with the any-node policy. To use the leaf-node policy
# please replace the policy "-p any " with "-p all"

cd ./sample-binaries/
for module in ${binary_modules[@]};do
  echo "patching module $module using any-node policy to ${module}.any"
  $bcov_bin_path -m patch -p any -v 1 -i "${module}" -o "${module}.any"
done

echo ""
echo "collecting coverage data for a sample run of perl"
chmod u+x ./perl.any
export LD_PRELOAD="$bcov_rt_lib_path"
./perl.any -e 'print "Hello, bcov\n"'


echo "reporting coverage data of the previous run to file: ${PWD}/report.out"
$bcov_bin_path -m report -p any -i ./perl -d perl.any.*.bcov > report.out

# Clean up to repeat the experiment
rm *.bcov
