function [value, isterminal, direction] = event_switch(~,x)
    value = x(1) + x(3); % switching occurs when x_3 = -x_1
    isterminal = 1; % stop integration
    direction = 0; % detect all zero crossings
end

