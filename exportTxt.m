dlmwrite('X_train.txt',X_train','precision',16)
dlmwrite('Y_train.txt',Y_train','precision',16)
dlmwrite('sigmaF.txt',sigf','precision',16)
dlmwrite('sigmaN.txt',sign','precision',16)

ls = ls(1:end);
dlmwrite('ls.txt',ls','precision',16)
dlmwrite('X_test.txt',X_test','precision',16)
dlmwrite('Y_test.txt',Y_test','precision',16)

dlmwrite('X_train0.txt',X_train(1:100,:)','precision',16)
dlmwrite('Y_train0.txt',Y_train(1:100,:)','precision',16)