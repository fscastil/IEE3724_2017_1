function [M, Xn, d] = faces_lh(first_face, last_face, first_sample, last_sample)
  M = [];
  Mx = [];
  f=Bio_statusbar('Beginning extraction ...');
  total = (last_face - first_face + 1) * (last_sample - first_sample + 1);

  for face=first_face:last_face
    for sample=first_sample:last_sample
      disp(['Img ' num2str(face) ' sample ' num2str(sample)]);
      % f = Bio_statusbar(title, f);
      im = ['faces/face_' num2fixstr(face, 4) '_' num2fixstr(sample, 4) '.png'];
      I = imread(im);
      [X, Xn] = double_lbp_haralick(I);
      M = [M; X];
      f = Bio_statusbar(((face - first_face) * (last_sample - first_sample + 1) + sample - first_sample + 1) / total, f);
    end
  end
  delete(f);
end
