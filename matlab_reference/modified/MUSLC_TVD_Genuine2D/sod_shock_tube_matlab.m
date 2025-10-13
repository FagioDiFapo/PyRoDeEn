%% Sod's Shock Tube MATLAB implementation for comparison with Python
% Performs Sod's shock tube test and saves execution times and solution data
% Matches the Python implementation for fair comparison

clear; close all; clc;
tic; % Start timing the whole benchmark

% Parameters
global gamma;
tEnd = 0.15;     % Final time - match Python version's tEnd
CFL = 0.5;       % CFL number
n = 5;           % Degrees of freedom
gamma = (n+2)/n; % Ratio of specific heats
plot_results = true; % Set to true to visualize the simulation

% Set up grid sizes
if plot_results
    % Use a higher resolution for better visualization
    nx = 200;
    ny = 1;  % Use a 1D configuration like Python
else
    % Use a range of grid sizes for benchmarking
    nx = 200;
    ny = 1;
end

fprintf('Running Sod shock tube simulation with grid size %dx%d...\n', nx, ny);

% Discretize spatial domain
Lx = 1.0; dx = Lx/nx; xc = dx/2:dx:Lx;
Ly = 0.1; dy = Ly/ny; yc = dy/2:dy:Ly;  % Thin domain for 1D-like behavior
[x, y] = meshgrid(xc, yc);

% Create Sod shock tube IC (similar to Python sod_ic_2d)
% Sod's values: rho=1, u=0, p=1 (left); rho=0.125, u=0, p=0.1 (right)
density = ones(size(x));
density(x >= 0.5) = 0.125;  % Right state
u_vel = zeros(size(x));
v_vel = zeros(size(x));
pressure = ones(size(x));
pressure(x >= 0.5) = 0.1;   % Right state

% Calculate energy
E0 = pressure./((gamma-1)*density) + 0.5*(u_vel.^2 + v_vel.^2);
c0 = sqrt(gamma*pressure./density);
Q0 = cat(3, density, density.*u_vel, density.*v_vel, density.*E0);

% Ghost cells
nx_g = nx + 2; ny_g = ny + 2;
q0 = zeros(ny_g, nx_g, 4);
q0(2:ny_g-1, 2:nx_g-1, 1:4) = Q0;

% Boundary Conditions in ghost cells (transmissive)
q0(:, 1, :) = q0(:, 2, :);
q0(:, nx_g, :) = q0(:, nx_g-1, :);
q0(1, :, :) = q0(2, :, :);
q0(ny_g, :, :) = q0(ny_g-1, :, :);

% Compute exact solution for comparison
% Make sure EulerExact.m is in the MATLAB path
addpath('../MUSCL_TVD');  % Adjust if needed to point to where EulerExact.m is
xe = linspace(0, Lx, 200);  % Fine grid for exact solution
[xe, re, ue, pe, ee, te, Me, se] = EulerExact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, tEnd, n);
Ee = ee + 0.5 * ue.^2;  % Total energy

% Initialize array for solution data
mid_row = ceil(ny/2);  % Middle row for 1D slice
solution_data = struct();
solution_data.x = xc;
solution_data.exact_x = xe;
solution_data.exact_density = re;
solution_data.exact_velocity = ue;
solution_data.exact_pressure = pe;
solution_data.exact_energy = Ee;

% Discretize time domain
vn = sqrt(u_vel.^2 + v_vel.^2);
lambda1 = vn + c0;
lambda2 = vn - c0;
a0 = max(abs([lambda1(:); lambda2(:)]));
dt = CFL * min(dx/a0, dy/a0);

% Start timing
start_time = toc;

% Load IC
q = q0; t = 0; it = 0;

% Internal indexes for removing ghost cells later
in = 2:ny_g-1;
jn = 2:nx_g-1;

% Set up plotting if enabled
if plot_results
    figure(1);

    % Extract initial values along middle row
    r_slice = q(mid_row+1, jn, 1);
    u_slice = q(mid_row+1, jn, 2)./r_slice;
    v_slice = q(mid_row+1, jn, 3)./r_slice;
    E_slice = q(mid_row+1, jn, 4)./r_slice;
    p_slice = (gamma-1) * r_slice .* (E_slice - 0.5 * (u_slice.^2 + v_slice.^2));

    % Setup 2x2 subplot
    subplot(2,2,1);
    h1 = plot(xc, r_slice, 'b.', xe, re, 'k-');
    xlabel('x'); ylabel('\rho');
    title('Density');
    legend('Numerical', 'Exact');
    grid on;

    subplot(2,2,2);
    h2 = plot(xc, u_slice, 'r.', xe, ue, 'k-');
    xlabel('x'); ylabel('u');
    title('Velocity');
    grid on;

    subplot(2,2,3);
    h3 = plot(xc, p_slice, 'm.', xe, pe, 'k-');
    xlabel('x'); ylabel('p');
    title('Pressure');
    grid on;

    subplot(2,2,4);
    h4 = plot(xc, E_slice, 'g.', xe, Ee, 'k-');
    xlabel('x'); ylabel('E');
    title('Energy');
    grid on;

    sgtitle(['Sod''s Shock Tube, t = ', num2str(t, '%.3f')]);
    drawnow;
end

% Time stepping
while t < tEnd
    % RK2 1st step
    res = MUSCL_EulerRes2d_v0(q, dt, dx, dy, nx_g, ny_g, 'MC', 'HLLE1d');
    qs = q - dt * res;

    % Apply BCs
    qs(:, 1, :) = qs(:, 2, :);
    qs(:, nx_g, :) = qs(:, nx_g-1, :);
    qs(1, :, :) = qs(2, :, :);
    qs(ny_g, :, :) = qs(ny_g-1, :, :);

    % RK2 2nd step / update q
    res2 = MUSCL_EulerRes2d_v0(qs, dt, dx, dy, nx_g, ny_g, 'MC', 'HLLE1d');
    q = 0.5 * (q + qs - dt * res2);

    % Apply BCs again
    q(:, 1, :) = q(:, 2, :);
    q(:, nx_g, :) = q(:, nx_g-1, :);
    q(1, :, :) = q(2, :, :);
    q(ny_g, :, :) = q(ny_g-1, :, :);

    % Compute flow properties for time step
    r = q(:,:,1);
    u = q(:,:,2)./r;
    v = q(:,:,3)./r;
    E = q(:,:,4)./r;
    p = (gamma-1) * r .* (E - 0.5 * (u.^2 + v.^2));
    c = sqrt(gamma * p ./ r);

    % Update dt and time
    vn = sqrt(u.^2 + v.^2);
    lambda1 = vn + c;
    lambda2 = vn - c;
    a = max(abs([lambda1(:); lambda2(:)]));
    dt = CFL * min(dx/a, dy/a);
    if t + dt > tEnd
        dt = tEnd - t;
    end
    t = t + dt;
    it = it + 1;

    % Update plot every few iterations if plotting is enabled
    if plot_results && mod(it, 10) == 0
        % Extract values along middle row
        r_slice = q(mid_row+1, jn, 1);
        u_slice = q(mid_row+1, jn, 2)./r_slice;
        v_slice = q(mid_row+1, jn, 3)./r_slice;
        E_slice = q(mid_row+1, jn, 4)./r_slice;
        p_slice = (gamma-1) * r_slice .* (E_slice - 0.5 * (u_slice.^2 + v_slice.^2));

        % Update plots
        set(h1(1), 'YData', r_slice);
        set(h2(1), 'YData', u_slice);
        set(h3(1), 'YData', p_slice);
        set(h4(1), 'YData', E_slice);
        sgtitle(['Sod''s Shock Tube, t = ', num2str(t, '%.3f')]);
        drawnow;
    end
end

% Record execution time
execution_time = toc - start_time;
fprintf('Simulation completed in %.6f seconds\n', execution_time);

% Extract final solution along middle row for saving
r_final = q(mid_row+1, jn, 1);
u_final = q(mid_row+1, jn, 2)./r_final;
v_final = q(mid_row+1, jn, 3)./r_final;
E_final = q(mid_row+1, jn, 4)./r_final;
p_final = (gamma-1) * r_final .* (E_final - 0.5 * (u_final.^2 + v_final.^2));

% Store numerical solution in structure
solution_data.numerical_density = r_final;
solution_data.numerical_velocity = u_final;
solution_data.numerical_pressure = p_final;
solution_data.numerical_energy = E_final;
solution_data.execution_time = execution_time;

% Save solution data to file
save('matlab_sod_solution.mat', 'solution_data');

% Save numerical data to CSV for easy comparison with Python

% Save individual variables to separate files (similar to Python output format)
dlmwrite('x_coords.txt', xc', 'delimiter', ',', 'precision', '%.10f');
dlmwrite('density.txt', r_final', 'delimiter', ',', 'precision', '%.10f');
dlmwrite('velocity.txt', u_final', 'delimiter', ',', 'precision', '%.10f');
dlmwrite('pressure.txt', p_final', 'delimiter', ',', 'precision', '%.10f');
dlmwrite('energy.txt', E_final', 'delimiter', ',', 'precision', '%.10f');

% Also save as a single CSV for easier viewing
fid = fopen('matlab_sod_results.txt', 'w');
fprintf(fid, 'MATLAB Sod Shock Tube Results (t = %.3f)\n', tEnd);
fprintf(fid, 'x,density,velocity,pressure,energy\n');
for i = 1:length(xc)
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        xc(i), r_final(i), u_final(i), p_final(i), E_final(i));
end
fclose(fid);

% Final comparison plot
if plot_results
    figure(2);

    subplot(2,2,1);
    plot(xc, r_final, 'b.', xe, re, 'k-');
    xlabel('x'); ylabel('\rho');
    title('Density');
    legend('MATLAB Numerical', 'Exact');
    grid on;

    subplot(2,2,2);
    plot(xc, u_final, 'r.', xe, ue, 'k-');
    xlabel('x'); ylabel('u');
    title('Velocity');
    grid on;

    subplot(2,2,3);
    plot(xc, p_final, 'm.', xe, pe, 'k-');
    xlabel('x'); ylabel('p');
    title('Pressure');
    grid on;

    subplot(2,2,4);
    plot(xc, E_final, 'g.', xe, Ee, 'k-');
    xlabel('x'); ylabel('E');
    title('Energy');
    grid on;

    sgtitle(['Sod''s Shock Tube - Final Solution (t = ', num2str(tEnd, '%.3f'), ')']);

    % Save the final plot
    saveas(gcf, 'matlab_sod_results.png');
end

% Save parameters for reference
fid = fopen('parameters.txt', 'w');
fprintf(fid, 'nx = %d\n', nx);
fprintf(fid, 'ny = %d\n', ny);
fprintf(fid, 'tEnd = %.3f\n', tEnd);
fprintf(fid, 'CFL = %.3f\n', CFL);
fprintf(fid, 'gamma = %.3f\n', gamma);
fprintf(fid, 'iterations = %d\n', it);
fprintf(fid, 'execution_time = %.6f\n', execution_time);
fclose(fid);

% Display overall time
fprintf('Total script execution time: %.2f seconds\n', toc);
fprintf('Simulation time: %.3f, Iterations: %d\n', t, it);
fprintf('Results saved to current directory\n');
