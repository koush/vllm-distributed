FROM vllm/vllm-openai:latest-x86_64

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# needed for nvfp4
RUN pip3 install -U flashinfer-python==0.3.1

#RUN apt -y install openssh-server && mkdir -p /run/sshd && ssh-keygen -A
# enable permit root login
#RUN bash -c 'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config'
#RUN mkdir ~/.ssh && bash -c 'echo ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDLkgoPsRwFJFXmjiHeRzuCfarpatvyYzJHcXNBzPPEtZ657zPo3s42+eq9ng2TTVN2oqCs82MeBAx8+NC6rydn/uaiSJcRFdRJalWtNJtyNKWWwjiWWYx/ldunJBMFI+Iizv+I1BCiIa9fUdSq42sd8LhgIdiI2h1Uru0PiXxYo3XuViOAkZiuXaWyJltxoUy7sZSX80m2O8bYNH4tY/MRatKCA+x3hHYsrzbNJYMObyRe01+WUhOQ/bV9jiWxOCy3i0geskdQUsaXiSQoF4KCdCO5et51sORZLEkOVy7eaoaVA1wx+TbO2D1OkbPGWVfKTDlN5qEXu5jl4mWMY/jsxhd0mr9iY7/oQUInqAFOXPp0Z7By5KZVfj6z9Ks344qefugyzpZk1XxpnefpJixrv+SJu+8SxQXTdxqdYOvxnC5Ep7pXAq6OuDzxQR6pstO6ao/RQAshMbZX5/5grwmATLAtJjAv6ElZJNmPUIOIrddlg0wo6NYGgGGSj54m0j0= koush@Mac-Studio.localdomain >> /root/.ssh/authorized_keys'
