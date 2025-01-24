tmp_src_filename=_fdfs_check_bits_.c
cat <<EOF > $tmp_src_filename
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
int main()
{
        printf("%d\n", (int)sizeof(long));
        return 0;
}
EOF

gcc -D_FILE_OFFSET_BITS=64 -o a.out $tmp_src_filename
OS_BITS=`./a.out`

rm $tmp_src_filename a.out

TARGET_LIB="/usr/lib"
if [ "`id -u`" = "0" ]; then
  ln -fs $TARGET_LIB/libfastcommon.so.1 /usr/lib/libfastcommon.so
  ln -fs $TARGET_LIB/libfdfsclient.so.1 /usr/lib/libfdfsclient.so

  if [ "$OS_BITS" = "8" ]; then
     ln -fs $TARGET_LIB/libfastcommon.so.1 /usr/lib64/libfastcommon.so
     ln -fs $TARGET_LIB/libfdfsclient.so.1 /usr/lib64/libfdfsclient.so
  fi
fi

