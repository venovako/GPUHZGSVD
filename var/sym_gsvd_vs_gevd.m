function sym_gsvd_vs_gevd(g11)
% an example for a revision of the paper by Vedran Novakovic and Sanja Singer
  disp(sprintf('test with G(1,1)=%.15e',g11));
  FF=sym([1 1 1 1; 0 1 1 1; 0 0 1 1; 0 0 0 1]);
  disp(sprintf('matrix F, cond(F)=%.15e',cond(FF)));
  disp(sprintf(' %.15e',FF(1,:)));
  disp(sprintf(' %.15e',FF(2,:)));
  disp(sprintf(' %.15e',FF(3,:)));
  disp(sprintf(' %.15e',FF(4,:)));
  GG=FF;
  GG(1,1)=sym(g11,'f');
  disp(sprintf('matrix G, cond(G)=%.15e',cond(GG)));
  disp(sprintf(' %.15e',GG(1,:)));
  disp(sprintf(' %.15e',GG(2,:)));
  disp(sprintf(' %.15e',GG(3,:)));
  disp(sprintf(' %.15e',GG(4,:)));
  AA=FF'*FF;
  disp(sprintf('matrix A=F''*F, cond(A)=%.15e',cond(AA)));
  disp(sprintf(' %.15e',AA(1,:)));
  disp(sprintf(' %.15e',AA(2,:)));
  disp(sprintf(' %.15e',AA(3,:)));
  disp(sprintf(' %.15e',AA(4,:)));
  BB=GG'*GG;
  disp(sprintf('matrix B=G''*G, cond(B)=%.15e',cond(BB)));
  disp(sprintf(' %.15e',BB(1,:)));
  disp(sprintf(' %.15e',BB(2,:)));
  disp(sprintf(' %.15e',BB(3,:)));
  disp(sprintf(' %.15e',BB(4,:)));
  TT=inv(BB)*AA;
  EE=eig(TT);
  E=sort(double(EE));
  disp('eig(A,B)');
  disp(sprintf(' %.15e',E));
end
