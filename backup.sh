#!/bin/bash

if [ -z "$1" ];
	then
	echo "Please provide system name"
	read hname
else
	 hname="$1"
fi

echo "Test $hname"
ssh-keygen
ssh-copy-id -i ~/.ssh/id_rsa root@192.168.47.100
apt install pixz -y
apt install pv -y

#systemctl stop runpod
#systemctl stop docker.socket
#systemctl stop docker
ssh root@192.168.47.100 "mkdir /mnt/user/backup/$hname"
tar -c -I 'pixz -k -1' -f - /var/lib/docker | pv | ssh root@192.168.47.100 "cat > /mnt/user/backup/$hname/docker.tar.pixz"
