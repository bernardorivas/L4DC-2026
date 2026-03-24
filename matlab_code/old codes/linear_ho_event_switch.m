function [value, isterminal, direction] = linear_ho_event_switch(~,x)
    value = x(1); % switching occurs when x_1 = 0
    isterminal = 1; % stop integration
    direction = 0; % detect all zero crossings
end