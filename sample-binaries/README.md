# Sample binaries

This archive provides a sample of our subject binaries patched with the bcov tool.
Patched binaries have the suffix `.any` to indicate that we used the any-node policy.
The binaries were compiled with `gcc-7.4` in release build.

In the following we give a few quick tutorials to better understand the instrumentation
techniques implemented in our tool. Let's pick the binary `gas` as our running example.


## Compare loadable segments

The following commands show the loadable segments in `gas` and `gas.any`

```sh
readelf -l gas
readelf -l gas.any
```

Comparing the outputs, you will notice two additional loadable segments in the patched
binary. The first segment holds trampoline code while the second segment stores
coverage data. Full segments details are available in the following listing. We
attached the `readelf` table header for convenience.

```
Type           Offset             VirtAddr           PhysAddr
               FileSiz            MemSiz              Flags  Align

 LOAD           0x00000000001d9000 0x00000000005c4000 0x00000000005c4000
                0x0000000000072ba0 0x0000000000072ba0  R E    0x1000
 LOAD           0x000000000024c000 0x0000000000637000 0x0000000000637000
                0x0000000000007813 0x0000000000007813  RW     0x1000

```

## Coverage data format

Let's now examine the header of the data segment which is located at file offset
`0x000000000024c000`. The header size is 24 bytes and the remaining part of the segment
is a byte array used to track coverage data. Running the following command,

```
 xxd -p -l 24 -s 0x000000000024c000 gas.any
```

will dump the header as a hex array which we split into separate fields.

```
2e42434f56555555 ffffffffffffffff fa770000 00000000
```

The details of the header format are:
- 8 byte. Magic number required to identify `bcov` data segments.
- 8 byte. Stores the base address of the module. It enables reporting the actual
  address of position-independent modules. This field is filled by our run-time
  library `libbcov-rt`.
- 4 byte. Size of the byte array (number of probes). For this particular binary
  it is 0x77fa (30714 probes).
- 4 byte. Process ID.

## Compare code patch


You can use `objdump` to disassemble the `.text` section in both binaries and compare
instructions.

```sh
objdump -Mintel -j.text -d gas
objdump -Mintel -j.text -d gas.any
```

This can be tedious to compare. A quick way to figure out where the patches are
located is to search for branches marked by objdump as targeting an address
beyond the end of the code section, i.e., branches to trampoline code.

```sh
objdump -Mintel -j.text -d gas.any | grep '<_end+'
```

Let's focus our attention on the first of these probes located in BB at `0x403780`.
We can run the following commands and compare the outputs.

```sh
objdump -Mintel -j.text -d gas | grep -P '\s403780:' -A4
objdump -Mintel -j.text -d gas.any | grep -P '\s403780:' -A4
```

The original code is here

```
403780:	49 89 34 24  	mov    QWORD PTR [r12],rsi
403784:	49 83 c4 08  	add    r12,0x8
403788:	48 ff c0     	inc    rax
40378b:	48 83 c6 20  	add    rsi,0x20
40378f:	eb e4        	jmp    403775 <elf_create_symbuf+0x35>
```

You can find significant changes after patching with `bcov`. The original BB
at `0x403780` consists of only two instructions both of which are 4 byte size.
Therefore, both of them had to be relocated to the trampoline and replaced with
a detour (5 bytes) stargeting the address `0x5c4000`


```
403780:	90                   	nop
403781:	90                   	nop
403782:	90                   	nop
403783:	e9 78 08 1c 00       	jmp    5c4000 <_end+0x570>
403788:	48 ff c0             	inc    rax
```

Now we examine the trampoline. However, we need to use `gdb` this time since `objdump`
does not seem to support disassembly at arbitrary code addresses.

```sh
gdb -batch gas.any \
     -ex "set disassembly-flavor intel" -ex "break main" -ex "run" \
     -ex "disassemble 0x5c4000,0x5c4020" \
     -ex quit
 ```

And here we have the trampoline located at `0x5c4000`. It starts by updating
coverage data and followed by the two relocated instructions. Then, control
flow will be restored back to address `0x403788` which immediately follows
the original BB.

 ```
 Dump of assembler code from 0x5c4000 to 0x5c4020:
   0x00000000005c4000:	mov    BYTE PTR [rip+0x73011],0x1        # 0x637018
   0x00000000005c4007:	mov    QWORD PTR [r12],rsi
   0x00000000005c400b:	add    r12,0x8
   0x00000000005c400f:	jmp    0x403788 <elf_create_symbuf+72>
   0x00000000005c4014:	mov    BYTE PTR [rip+0x72ffe],0x1        # 0x637019
   0x00000000005c401b:	add    rsi,0x20
   0x00000000005c401f:	jmp    0x403775 <elf_create_symbuf+53>
```
