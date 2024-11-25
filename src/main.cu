#include "kernels.cuh"
#include "utils.cuh"
#include "constants.cuh"

int main() {
    initializeConstants(); // Função que inicializa constantes na memória da GPU.
    float *h_rho, *h_phi, *d_rho, *d_phi;
    allocateMemory(h_rho, h_phi, d_rho, d_phi);
    initializeHostArrays(h_rho, h_phi);

    copyToDevice(h_rho, h_phi, d_rho, d_phi);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // JUNTAR TUDO EM UM KERNEL SÓ. SALVAR VALORES DE PHI A CADA PASSO E VISUALIZAR

    for (int t = 0; t < nsteps; t++) {
        // placeholder pro começo

        momentCalc<<<numBlocks, threadsPerBlock>>>(
            d_f, d_rho, d_ffx, d_ffy, d_ffz,
            d_cix, d_ciy, d_ciz,
            d_ux, d_uy, d_uz, d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            nx, ny, nz, fpoints, cssq, d_w
        );

        collisionCalc<<<numBlocks, threadsPerBlock>>>(
            d_f, d_g, d_phi, d_rho, d_ux, d_uy, d_uz,
            d_ffx, d_ffy, d_ffz, d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz,
            d_cix, d_ciy, d_ciz, d_w, d_w_g,
            d_normx, d_normy, d_normz,
            nx, ny, nz, fpoints, gpoints, cssq, omega, sharp_c
        );

        // placeholder pro final
    }

    freeMemory(h_rho, h_phi, d_rho, d_phi);
    return 0;
}
