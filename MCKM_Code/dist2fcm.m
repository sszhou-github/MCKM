function out = dist2fcm(center, data)
%DISTFCM Distance measure in fuzzy c-mean clustering.
%	OUT = DISTFCM(CENTER, DATA) calculates the  Euclidean distance
%	between each row in CENTER and each row in DATA, and returns a
%	distance matrix OUT of size M by N, where M and N are row
%	dimensions of CENTER and DATA, respectively, and OUT(I, J) is
%	the distance between CENTER(I,:) and DATA(J,:).

% out = sqrt(sum(center.^2,2)+sum(data.^2,2)' - 2*center*data');
out = sqrt(sum(center.^2,2)+sum(data.^2,2)' - 2*center*data');

end
