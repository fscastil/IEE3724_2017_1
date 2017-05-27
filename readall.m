function [M, Mn, d] = readall(path)
  files = dir(path);
  M = [];
  Mn = [];
  d = [];
  f=Bio_statusbar('Extracting ...');
  total = numel(files);
  advance = 0;

  for i=1:numel(files)
    sample = str2num(files(i).name(11:14));
    face = str2num(files(i).name(6:9));
    disp(['Img ' num2str(face) ' sample ' num2str(sample)]);
    if (sample > 10)
      d = [d; face];
      I = imread([files(i).folder '/' files(i).name]);
      [X, Mn] = double_lbp_haralick(I);
      M = [M; X];
    end
    advance = advance + 1;
    f = Bio_statusbar(advance / total, f);
  end
  delete(f);
end
