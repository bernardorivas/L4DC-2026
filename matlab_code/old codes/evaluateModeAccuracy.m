function metrics = evaluateModeAccuracy(X_traj, surf, modeGT_traj)
% Predict modes on a tested trajectory using sign of learned surface, then
% compute accuracy / precision / recall / F1 / mIoU if GT labels provided.
%
% Inputs:
%   X_traj      : (T+1)×n states along the tested trajectory
%   surf        : struct with fields .a, .kappa
%   modeGT_traj : (optional) (T+1)×1 ground-truth mode labels in {1,2}
%
% Output:
%   metrics: struct with fields accuracy, precision_macro, recall_macro,
%            f1_macro, miou, confusion (empty if GT not provided)

    Phi = buildMonomialMatrix(X_traj, surf.kappa);
    f   = Phi * surf.a;
    mode_pred = ones(size(f));
    mode_pred(f < 0) = 2;

    metrics = struct('accuracy',NaN,'precision_macro',NaN,'recall_macro',NaN, ...
                     'f1_macro',NaN,'miou',NaN,'confusion',[]);

    if nargin >= 3 && ~isempty(modeGT_traj)
        cm = confusionmat(modeGT_traj(:), mode_pred(:));
        metrics.confusion = cm;
        metrics.accuracy  = sum(diag(cm))/sum(cm(:));

        prec = diag(cm) ./ max(sum(cm,2),1);
        rec  = diag(cm) ./ max(sum(cm,1)',1);
        f1   = 2*(prec.*rec) ./ max(prec+rec, eps);

        inter = diag(cm);
        union = sum(cm,2)+sum(cm,1)'-inter;
        miou  = mean(inter ./ max(union,1));

        metrics.precision_macro = mean(prec);
        metrics.recall_macro    = mean(rec);
        metrics.f1_macro        = mean(f1);
        metrics.miou            = miou;
    end
end
