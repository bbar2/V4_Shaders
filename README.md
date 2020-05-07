# V4_Shaders


Vulkan tutorial project based on Tutorial by Alexandar Overvoorde.

Linux Ubunto 19.10 eoan

CLion IDE with CMake compile system

The tutorial has mostly worked. One exception is the tutorial use of LD_LIBRARY_PATH did not work.

    Per the tutorial, I updated .bashrc to create path variables

Although this seemed to work, at run time, I received validation layer errors, something like: "..so cannot open shared object file ..".

    It turns out that since Ubunto 9.04, LD_LIBRARY_PATH cannot be set in .profile or .bashrc files this way.

    You must use /etc/ld.so.conf.d/*.conf files.
    Basically create your own *.conf file with the path, then run ldconfig.

    Here is what I did:
```sh    
    sudo cd /etc/ld.so.conf.d
    sudo vim vulkan_sdk_path.conf !make up a name that ends in .conf
```    
    Add the following line to the new vulkan_sdk_path.conf file
    /home/barry/Vulkan/1.2.131.2/x86_64/lib !of course use your SDK path

    sudo ldconfig

    This eliminated the "cannot load shared object file" error.

