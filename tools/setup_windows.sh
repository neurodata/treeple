PLATFORM=$(PYTHONPATH=tools python -c "import openblas_support; print(openblas_support.get_plat())")

printenv

# The 32/64 bit Fortran wheels are currently coming from different locations.
if [[ $PLATFORM == 'win-32' ]]; then
    # 32-bit openBLAS
    # Download 32 bit openBLAS and put it into c/opt/32/lib
    unzip $target -d /c/opt/
    cp /c/opt/32/bin/*.dll /c/opt/openblas/openblas_dll
    # rm /c/opt/openblas/if_32/32/lib/*.dll.a
else
    # 64-bit openBLAS
    unzip $target -d /c/opt/
    cp /c/opt/64/bin/*.dll /c/opt/openblas/openblas_dll
fi