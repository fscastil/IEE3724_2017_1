function [X, Xn] = double_lbp_haralick(I)
  % H = [
  %     'Angular Second Moment  '
  %     'Contrast               '
  %     'Correlacion            '
  %     'SumOSquares            '
  %     'InverseDiffMoment      '
  %     'SumAverage             '
  %     'SumEntropy             '
  %     'SumVariance            '
  %     'Entropy                '
  %     'DifferenceVariance     '
  %     'DifferenceEntropy      '
  %     'InfoMeasuresCorrel1    '
  %     'InfoMeasuresCorrel2    '
  %     'MaxCorrelCoefficient   '
  % ];
  uniform_options.vdiv = 1;
  uniform_options.hdiv = 1;
  uniform_options.mappingtype = 'u2';
  ri_options.vdiv = 1;
  ri_options.hdiv = 1;
  ri_options.mappingtype = 'ri';
  options.dharalick = 1:5;

  gabor_options.Lgabor = 8;
  gabor_options.Sgabor = 8;
  gabor_options.Mgabor = 5;
  gabor_options.fhgabor = 2;                 % highest frequency of interest
  gabor_options.flgabor = 0.1;
  gabor_options.show = 0;

  X = []; Xn = [];
  % GET UNIFORM LBP
  % [X, Xn] = Bfx_lbp(I, uniform_options);
  % GET RI LBP
  % [Y, Yn] = Bfx_lbp(I, ri_options);
  % X = [X Y];
  % Xn = [Xn; Yn];
  % WINDOW FOR HARALICK
  [rows, columns] = size(I);
  rstep = ceil(rows/4); cstep = ceil(columns/4); % In case of non divisibility I truncate the segment
  for row=1:rstep:(rows - 1)
    for column=1:cstep:(columns - 1)
      J = I(row: min(row + rstep - 1, rows), column: min(column + cstep - 1, columns));
      [Y, Yn] = Bfx_haralick(J, options);
      X = [X Y];
      Xn = [Xn; Yn];
      [Y, Yn] = Bfx_lbp(J, uniform_options);
      X = [X Y];
      Xn = [Xn; Yn];
      [Y, Yn] = Bfx_lbp(J, ri_options);
      X = [X Y];
      Xn = [Xn; Yn];
      [Y, Yn] = Bfx_gabor(J, gabor_options);
      X = [X Y];
      Xn = [Xn; Yn];

      % GABOR
      % LBP
      %
      Xn = [Xn; Yn];
    end
  end
end
