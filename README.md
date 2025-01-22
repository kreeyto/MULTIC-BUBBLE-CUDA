# Bubble-GPU

Bubble-GPU é um projeto para simulações de fluidos usando o método Lattice Boltzmann (LBM), implementado com suporte para GPUs, permitindo a execução eficiente de simulações computacionalmente intensivas. O foco atual está na simulação de bolhas estacionárias e oscilantes em 3D utilizando os modelos D3Q19 e D3Q27.

## Estrutura do Projeto

- **src/**: Contém o código-fonte principal, incluindo kernels CUDA e scripts de compilação.
- **bin/**: Diretório de saída para os binários compilados.
- **matlabFiles/**: Contém scripts MATLAB para análise e visualização.
- **post/**: Scripts para pós-processamento dos resultados da simulação.
- **pipeline.sh**: Script principal para executar o pipeline de compilação, simulação e pós-processamento.

## Como Executar

1. **Compilar e Executar a Simulação**:

   Use o script `pipeline.sh` para compilar e executar o simulador:

   ```bash
   ./pipeline.sh <fluid_model> <phase_model> <id> <save_binary>
   ```

   - **`<fluid_model>`**: Modelo de fluido (exemplo: `FD3Q19`).
   - **`<phase_model>`**: Modelo de fase (exemplo: `PD3Q15`).
   - **`<id>`**: Identificador único para a simulação (exemplo: `000`).
   - **`<save_binary>`**: Defina `0` para compilar e executar sem salvar binários (para visualização em MATLAB) ou `1` 		para salvar (para visualização em Paraview).

   Exemplo:

   ```bash
   ./pipeline.sh FD3Q19 PD3Q15 001 1
   ```

2. **Resultados**:
   - Saídas de simulação serão salvas em:
     - `bin/<fluid_model>_<phase_model>/<id>/` (se `save_binary=1`).
     - `matlabFiles/<fluid_model>_<phase_model>/<id>/` (se `save_binary=0`).

3. **Pós-Processamento**:
   Caso `save_binary=1`, o script também executará o pós-processamento automaticamente.

## Exemplos de Execução

- **Simulação sem salvar binários**:
  ```bash
  ./pipeline.sh FD3Q19 PD3Q15 001 0
  ```

- **Simulação salvando binários**:
  ```bash
  ./pipeline.sh FD3Q19 PD3Q15 002 1
  ```

