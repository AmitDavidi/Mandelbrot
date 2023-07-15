# Mandelbrot-threading-AVX256
Exploring the Mandelbrot set, Brute force processing with Threading and AVX256.

## Quick tutorial
Q - Increase Iterations
W - Decrease Iterations

Left click - Zoom in

Right click - go back

Middle Click [HOLD and drag] - move

a - Zoom continously 

s - zoom out continously

## How to run

# If you don't have MSYS2 and MinGW (And you can't run cpp programs) Follow the instructions of this video : `https://www.youtube.com/watch?v=jnI1gMxtrB4`
a summary:
* Install MSYS2
- Open MSYS2 Terminal and run the following commands (Note - press y, and enter when asked to):
1. `pacman -Syu`
2. `pacman -Su`
3. For a 64bit system: `pacman -S mingw-w64-x86_64-gcc`, for a 32bit system: `pacman -S mingw-w64-i686-gcc`
4. For a 64 bit system: `pacman -S mingw-w64-x86_64-make` for a 32bit system: `pacman -S mingw-w64-i686-make`
5. Configure the PATH variable to include the path of gcc, in my computer its `C:\msys64\mingw64\bin` (Add that to your path variable)

# Compilation
* Compile the program - Note I added `-O3`, you don't have to :) : `gcc -o Mandlebrot.exe main.cpp -luser32 -lgdi32 -lopengl32 -lgdiplus -lShlwapi -ldwmapi -lstdc++ -static -std=c++17 -mavx2 -mfma -O3`
* run it ` .\Mandlebrot.exe `

### Enjoy your journy into the infinite :)
