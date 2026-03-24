clear; clc; close all;

% parameters
mu1 = 0.5;   % negative damping inside strip (since f = -mu1)
mu2 = 1.0;   % positive damping outside strip (since f = +mu2)
A_in  = [0 1; -1  mu1];
A_out = [0 1; -1 -mu2];

% region test and (optional) tiny hysteresis
hyst = 0;                           % can be 0; small >0 helps avoid chatter
in_strip = @(x) (abs(x(1)) <= 1 - hyst);

% event: hit either x1=+1 or x1=-1
function [value,isterminal,direction] = switch_events(~,x)
    value      = [x(1)-1; x(1)+1];     % zeros at +1 and -1
    isterminal = [1; 1];               % stop integration
    direction  = [0; 0];               % detect all crossings
end
options = odeset('Events', @switch_events, 'RelTol',1e-8, 'AbsTol',1e-10);

% simulate many trajectories
n_samples = 200;
t_log = cell(1,n_samples); y_log = cell(1,n_samples);
dy_log = cell(1,n_samples); mode_log = cell(1,n_samples);

for k = 1:n_samples
    x0 = (rand(2,1)-0.5)*8;           % IC in [-4,4]^2
    t0 = 0; tf = 10;
    while t0 < tf
        if in_strip(x0), A = A_in; mode = 1; else, A = A_out; mode = 2; end
        % integrate current linear mode until switch
        [t_part, y_part, te, xe] = ode45(@(t,x) A*x, [t0 tf], x0, options);
        dy_part   = (A * y_part.').';
        mode_part = mode*ones(size(t_part));

        t_log{k}   = [t_log{k}; t_part];
        y_log{k}   = [y_log{k}; y_part];
        dy_log{k}  = [dy_log{k}; dy_part];
        mode_log{k}= [mode_log{k}; mode_part];

        if isempty(te), break; end
        % restart exactly at the hit point
        t0 = te(end);
        x0 = xe(end,:).';
        % tiny push to the new side to avoid re-detecting the same event
        x0(1) = x0(1) + sign(x0(2))*1e-8; % since traj is clockwise, push slightly to the right if x2>0 and vice versa
    end
end

% pack data
data.y    = cell2mat(y_log');
data.dy   = cell2mat(dy_log');
data.mode = cell2mat(mode_log');
save("switching_limit_cycle_data.mat", "data");

% plots
figure(1); clf; hold on;
for k=1:n_samples
    plot(y_log{k}(:,1), y_log{k}(:,2), '-');
end
xline( 1,'k--'); xline(-1,'k--'); axis equal
xlabel('$x_1$', 'Interpreter','latex'); ylabel('$x_2$', 'Interpreter','latex'); title('Phase Portrait with switching at $|x_1|=1$','Interpreter','latex');

figure(2); clf; hold on;
for k=1:n_samples, stairs(t_log{k}, mode_log{k}, '-'); end
xlabel('Time', 'Interpreter','latex'); ylabel('Mode', 'Interpreter','latex'); title('Mode Switching', 'Interpreter','latex');

% "energy"-like diagnostic (omega=1)
omega = 1;
figure(3); clf;
for k=1:n_samples
    E = 0.5*y_log{k}(:,2).^2 + 0.5*omega^2*y_log{k}(:,1).^2;
    subplot(3,1,1); hold on; plot(y_log{k}(:,1), y_log{k}(:,2), '-');
    subplot(3,1,2); hold on; plot(t_log{k}, y_log{k});
    subplot(3,1,3); hold on; plot(t_log{k}, E);
end
subplot(3,1,1); xlabel('x_1'); ylabel('x_2'); title('Trajectories');
subplot(3,1,2); xlabel('t'); ylabel('x');
subplot(3,1,3); xlabel('t'); ylabel('Energy'); set(gca,'YScale','log');
