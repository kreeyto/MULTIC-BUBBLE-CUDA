clc; clearvars; close all

% ================================================================== %

% ALTERAR VARIAVEIS BASEADO NA SIMULAÇÃO EXECUTADA!!!!!!!
% Futuramente pretendo automatizar a definição dos parâmetros/diretórios baseado na simulação.

sim_dir = 'COLOQUE/O/DIRETORIO/AQUI'; % Substitua pelo diretório onde estão os arquivos .txt
file_prefix = '000_phi'; % Prefixo dos arquivos. Por padrão é <id>_phi
num_steps = 50; % Número de passos de simulação
res = 1; % Valor da resolução (definido em Bubble-GPU/src/var.cu)
nx = round(150 * res); % Substitua pelo valor correto de nx
ny = round(150 * res); % Substitua pelo valor correto de ny
nz = round(150 * res); % Substitua pelo valor correto de nz
% ================================================================== %

for t = 1:num_steps
    filename = sprintf('%s%s%d.txt', sim_dir, file_prefix, t);
    if ~isfile(filename)
        fprintf('Arquivo não encontrado: %s\n', filename);
        continue;
    end
    fprintf('Lendo arquivo: %s\n', filename);
    data = load(filename);
    
    total_elements = numel(data);
    expected_elements = nx * ny * nz;
    if total_elements ~= expected_elements
        error('Erro: O número de elementos no arquivo (%d) não corresponde ao esperado (%d).', ...
              total_elements, expected_elements);
    end
    
    phi = reshape(data, [nx, ny, nz]);
    
    slice_idx = round(ny/2); 
    imagesc(squeeze(phi(:, slice_idx, :)));
    colorbar;
    title(sprintf('Simulação passo %d', t));
    
    % Delay opcional para visualização
    % pause(0.1);
end
