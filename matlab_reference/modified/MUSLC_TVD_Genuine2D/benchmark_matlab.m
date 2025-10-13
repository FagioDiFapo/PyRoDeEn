%% Benchmark MATLAB implementation for comparison with Python vectorized code
% Performs blast test on increasing grid sizes and saves execution times
% Matches the Python benchmarking approach for fair comparison

clear; close all; clc;
tic; % Start timing the whole benchmark

% Parameters
global gamma;
tEnd = 0.3;      % Final time
CFL = 0.5;       % CFL number
n = 5;           % Degrees of freedom
gamma = (n+2)/n; % Ratio of specific heats
plot_results = false; % Set to false for benchmarking, true for visualization

% Benchmark configurations - grid sizes to test
if plot_results
    % If plotting is enabled, just use a single grid size for visualization
    grid_sizes = [50]; % Use a moderate size for visualization
else
    % Otherwise run the full benchmark suite
    grid_sizes = [20, 40, 60, 80, 90, 100, 110, 120, 130, 140]; % Match Python benchmark sizes
end
num_configs = length(grid_sizes);
execution_times = zeros(num_configs, 1);

% Open file for writing results
fid = fopen('matlab_benchmark_results.txt', 'w');
fprintf(fid, 'MATLAB Benchmark Results\n');
fprintf(fid, 'Grid Size,Execution Time (s)\n');

% Run benchmark for each grid size
for config = 1:num_configs
    nx = grid_sizes(config);
    ny = grid_sizes(config);

    fprintf('Running benchmark for grid size %dx%d...\n', nx, ny);

    % Discretize spatial domain - match Python version's domain
    Lx = 1.0; dx = Lx/nx; xc = dx/2:dx:Lx;
    Ly = 1.0; dy = Ly/ny; yc = dy/2:dy:Ly;
    [x, y] = meshgrid(xc, yc);

    % Set up blast IC (matching Python implementation)
    % Using center_rel_x = center_rel_y = 0.5 (centered blast)
    center_rel_x = 0.5;
    center_rel_y = 0.5;
    center_x = center_rel_x * Lx;
    center_y = center_rel_y * Ly;
    r0 = 0.1; % radius
    p_high = 1.0;
    p_low = 0.1;

    % Create blast IC (similar to Python blast_ic_2d)
    r = sqrt((x - center_x).^2 + (y - center_y).^2);
    density = ones(size(x));
    u_vel = zeros(size(x));
    v_vel = zeros(size(x));
    pressure = p_low * ones(size(x));
    pressure(r < r0) = p_high;

    % Calculate energy
    E0 = pressure./((gamma-1)*density) + 0.5*(u_vel.^2 + v_vel.^2);
    c0 = sqrt(gamma*pressure./density);
    Q0 = cat(3, density, density.*u_vel, density.*v_vel, density.*E0);

    % Set q-array & adjust grid for ghost cells
    nx_g = nx + 2; ny_g = ny + 2;
    q0 = zeros(ny_g, nx_g, 4);
    q0(2:ny_g-1, 2:nx_g-1, 1:4) = Q0;

    % Boundary Conditions in ghost cells
    q0(:, 1, :) = q0(:, 2, :);
    q0(:, nx_g, :) = q0(:, nx_g-1, :);
    q0(1, :, :) = q0(2, :, :);
    q0(ny_g, :, :) = q0(ny_g-1, :, :);

    % Discretize time domain
    vn = sqrt(u_vel.^2 + v_vel.^2);
    lambda1 = vn + c0;
    lambda2 = vn - c0;
    a0 = max(abs([lambda1(:); lambda2(:)]));
    dt0 = CFL * min(dx/a0, dy/a0);

    % Start timing this grid size
    start_time = toc;

    % Load IC
    q = q0; t = 0; it = 0; dt = dt0;

    % Internal indexes for removing ghost cells later
    in = 2:ny_g-1;
    jn = 2:nx_g-1;

    % Setup plotting if enabled
    if plot_results
        figure(2);
        % Extract initial pressure field (interior)
        r_plot = q(in, jn, 1);
        u_plot = q(in, jn, 2)./r_plot;
        v_plot = q(in, jn, 3)./r_plot;
        E_plot = q(in, jn, 4)./r_plot;
        p_plot = (gamma-1) * r_plot .* (E_plot - 0.5 * (u_plot.^2 + v_plot.^2));

        % Create the visualization
        h = imagesc(xc, yc, p_plot);
        colorbar;
        axis equal;
        xlabel('x');
        ylabel('y');
        title(['2D Blast Wave: Pressure, t=', num2str(t, '%.3f')]);
        colormap('jet');
        caxis([0 p_high]); % Set color scale limits
        drawnow;
        pause(1); % Give time to observe initial condition
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
        if t + dt > tEnd;
            dt = tEnd - t;
        end
        t = t + dt;
        it = it + 1;

        % Update plot every few iterations if plotting is enabled
        if plot_results && mod(it, 5) == 0
            % Extract pressure field (interior)
            p_interior = p(in, jn);
            set(h, 'CData', p_interior);
            title(['2D Blast Wave: Pressure, t=', num2str(t, '%.3f')]);
            drawnow;
        end
    end

    % Record execution time for this grid size
    execution_time = toc - start_time;
    execution_times(config) = execution_time;

    % Write result to file
    fprintf(fid, '%d,%f\n', nx, execution_time);

    % Display result
    fprintf('Grid size %dx%d completed in %.6f seconds\n', nx, ny, execution_time);
end

% Close the output file
fclose(fid);

% Only create the benchmark plot if we're not in visualization mode
if ~plot_results
    % Create a simple plot of the results
    figure(1);
    plot(grid_sizes, execution_times, 'o-', 'LineWidth', 2);
    xlabel('Grid Size (N x N)');
    ylabel('Execution Time (s)');
    title('MATLAB Execution Time vs Grid Size');
    grid on;

    % Save the plot
    saveas(gcf, 'matlab_benchmark_plot.png');

    % Save the execution times array for later use
    save('matlab_benchmark_data.mat', 'grid_sizes', 'execution_times');
end

% Display overall benchmark time
fprintf('Benchmark complete! Total time: %.2f seconds\n', toc);
fprintf('Results saved to matlab_benchmark_results.txt\n');