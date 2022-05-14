
# [Reference](https://www.redhat.com/sysadmin/partitions-fdisk)

- Show partitions including unmounted ```sudo fdisk -l```
- Create new FS ```sudo mkfs.ext4 /dev/sda4```
- Create a mountpoint ```sudo mkdir /store```
- Mount to the mountpoint ```sudo mount /dev/sda4 /store```
- Make it accessible without root ```sudo chown -R $USER /store```

- To auto-mount on boot ```sudo blkid /dev/sda4```
- Add entry to /etc/fstab ```UUID=9f4fc683-5f7e-4b83-aa41-6123bef1c4a7      /store   ext4     defaults         0 0```