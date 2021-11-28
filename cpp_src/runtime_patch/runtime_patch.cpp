#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <iostream>
#include <jemalloc/jemalloc.h>
#include <link.h>
#include <pybind11/embed.h>
#include <sstream> //for std::stringstream
#include <stdexcept>
#include <string> //for std::string
#include <sys/auxv.h>
#include <sys/mman.h>

struct strtab {
  char *tab;
  ElfW(Xword) size;
};

struct jmpreltab {
  ElfW(Rela) * tab;
  ElfW(Xword) size;
};

struct symtab {
  ElfW(Sym) * tab;
  ElfW(Xword) entsz;
};

extern void *alloc_cpu(size_t nbytes);
extern void free_cpu(void *data);

/* Backup of the real malloc function */
static void *(*realalloc_cpu)(size_t) = nullptr;
static void (*realfree_cpu)(void *) = nullptr;

/* My local versions of the alloc_cpu functions */
static void *myalloc_cpu(size_t size);
static void myfree_cpu(void *ptr);

int64_t time_since_epoch() {
  auto t = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t.time_since_epoch())
      .count();
}

static uint64_t epoch = 1;
std::string get_stats() {
  // Update the statistics cached by mallctl.
  epoch += 1;
  size_t sz;
  sz = sizeof(epoch);
  jemallctl("epoch", &epoch, &sz, &epoch, sz);

  // Get basic allocation statistics.  Take care to check for
  // errors, since --enable-stats must have been specified at
  // build time for these statistics to be available.
  size_t allocated, active, metadata, resident, mapped, retained;
  std::ostringstream out;
  sz = sizeof(size_t);
  if (jemallctl("stats.allocated", &allocated, &sz, NULL, 0) == 0 &&
      jemallctl("stats.active", &active, &sz, NULL, 0) == 0 &&
      jemallctl("stats.metadata", &metadata, &sz, NULL, 0) == 0 &&
      jemallctl("stats.resident", &resident, &sz, NULL, 0) == 0 &&
      jemallctl("stats.mapped", &mapped, &sz, NULL, 0) == 0 &&
      jemallctl("stats.retained", &retained, &sz, NULL, 0) == 0) {
    out << allocated << "," << active << "," << metadata << "," << resident
        << "," << mapped << "," << retained;
  }
  return out.str();
}

constexpr size_t gAlignment = 64;
void *jemellac_alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  void *data;
  int err = jeposix_memalign(&data, gAlignment, nbytes);
  if (err != 0) {
    throw std::invalid_argument("ran out of memory");
  }
  if(const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
    std::cout << "alloc," << time_since_epoch() << "," << data << "," << get_stats() << std::endl;
  }

  return data;
}

void jemalloc_free_cpu(void *data) {
  jefree(data);
  if(const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
    std::cout << "free," << time_since_epoch() << "," << data << ","
              << get_stats() << std::endl;
  }
}

/*************/
/* ELF stuff */
/*************/
static const ElfW(Phdr) * get_phdr_dynamic(const ElfW(Phdr) * phdr,
                                           uint16_t phnum, uint16_t phentsize) {
  int i;

  for (i = 0; i < phnum; i++) {
    if (phdr->p_type == PT_DYNAMIC)
      return phdr;
    phdr = (ElfW(Phdr) *)((char *)phdr + phentsize);
  }

  return nullptr;
}

static const ElfW(Dyn) *
    get_dynentry(ElfW(Addr) base, const ElfW(Phdr) * pdyn, uint32_t type) {
  ElfW(Dyn) * dyn;

  for (dyn = (ElfW(Dyn) *)(base + pdyn->p_vaddr); dyn->d_tag; dyn++) {
    if (dyn->d_tag == type)
      return dyn;
  }

  return nullptr;
}

static struct jmpreltab get_jmprel(ElfW(Addr) base, const ElfW(Phdr) * pdyn) {
  struct jmpreltab table;
  const ElfW(Dyn) * dyn;

  dyn = get_dynentry(base, pdyn, DT_JMPREL);
  table.tab = (dyn == nullptr) ? nullptr : (ElfW(Rela) *)dyn->d_un.d_ptr;

  dyn = get_dynentry(base, pdyn, DT_PLTRELSZ);
  table.size = (dyn == nullptr) ? 0 : dyn->d_un.d_val;
  return table;
}

static struct symtab get_symtab(ElfW(Addr) base, const ElfW(Phdr) * pdyn) {
  struct symtab table;
  const ElfW(Dyn) * dyn;

  dyn = get_dynentry(base, pdyn, DT_SYMTAB);
  table.tab = (dyn == nullptr) ? nullptr : (ElfW(Sym) *)dyn->d_un.d_ptr;
  dyn = get_dynentry(base, pdyn, DT_SYMENT);
  table.entsz = (dyn == nullptr) ? 0 : dyn->d_un.d_val;
  return table;
}

static struct strtab get_strtab(ElfW(Addr) base, const ElfW(Phdr) * pdyn) {
  struct strtab table;
  const ElfW(Dyn) * dyn;

  dyn = get_dynentry(base, pdyn, DT_STRTAB);
  table.tab = (dyn == nullptr) ? nullptr : (char *)dyn->d_un.d_ptr;
  dyn = get_dynentry(base, pdyn, DT_STRSZ);
  table.size = (dyn == nullptr) ? 0 : dyn->d_un.d_val;
  return table;
}

static void *get_got_entry(ElfW(Addr) base, struct jmpreltab jmprel,
                           struct symtab symtab, struct strtab strtab,
                           const char *symname) {

  ElfW(Rela) * rela;
  ElfW(Rela) * relaend;

  relaend = (ElfW(Rela) *)((char *)jmprel.tab + jmprel.size);
  for (rela = jmprel.tab; rela < relaend; rela++) {
    uint32_t relsymidx;
    char *relsymname;
    relsymidx = ELF64_R_SYM(rela->r_info);
    relsymname = strtab.tab + symtab.tab[relsymidx].st_name;
    if (strcmp(symname, relsymname) == 0)
      return (void *)(base + rela->r_offset);
  }

  return nullptr;
}

static void patch_got(ElfW(Addr) base, const ElfW(Phdr) * phdr, int16_t phnum,
                      int16_t phentsize) {

  const ElfW(Phdr) * dphdr;
  struct jmpreltab jmprel;
  struct symtab symtab;
  struct strtab strtab;

  void *(**alloc_cpugot)(size_t);
  void (**free_cpugot)(void *);

  dphdr = get_phdr_dynamic(phdr, phnum, phentsize);
  jmprel = get_jmprel(base, dphdr);
  symtab = get_symtab(base, dphdr);
  strtab = get_strtab(base, dphdr);

  alloc_cpugot = static_cast<void *(**)(size_t)>(
      get_got_entry(base, jmprel, symtab, strtab, "_ZN3c109alloc_cpuEm"));
  free_cpugot = static_cast<void (**)(void *)>(
      get_got_entry(base, jmprel, symtab, strtab, "_ZN3c108free_cpuEPv"));

  if (alloc_cpugot != nullptr) {
//    printf("found alloc_cpu\n");
    void *page = (void *)((intptr_t)alloc_cpugot & ~(0x1000 - 1));
    mprotect(page, 0x1000, PROT_READ | PROT_WRITE);
    realalloc_cpu = *alloc_cpugot;
    //    *alloc_cpugot = myalloc_cpu;
    *alloc_cpugot = jemellac_alloc_cpu;
  }
  if (free_cpugot != nullptr) {
//    printf("found free_cpu\n");
    void *page = (void *)((intptr_t)free_cpugot & ~(0x1000 - 1));
    mprotect(page, 0x1000, PROT_READ | PROT_WRITE);
    realfree_cpu = *free_cpugot;
    //    *free_cpugot = myfree_cpu;
    *free_cpugot = jemalloc_free_cpu;
  }
}

static int callback(struct dl_phdr_info *info, size_t size, void *data) {
  uint16_t phentsize;

  if (std::string(info->dlpi_name).find("libc10.so") != std::string::npos) {
//    printf("Patching GOT entry of \"%s\"\n", info->dlpi_name);
    phentsize = getauxval(AT_PHENT);
    patch_got(info->dlpi_addr, info->dlpi_phdr, info->dlpi_phnum, phentsize);
  }

  return 0;
}

/*****************/
/* Init function */
/*****************/
namespace py = pybind11;

__attribute__((destructor)) static void fini() {
  // Clean up
}

PyObject *python_alloc_cpu_fn_ptr;
PyObject *python_free_cpu_fn_ptr;

__attribute__((constructor)) static void init() {
//  printf("initing\n");
//  dl_iterate_phdr(callback, nullptr);

  //  auto runtime_mod = py::module_::import("runtime").attr("__dict__").ptr();
  //  auto runtime_mod_dict = py::reinterpret_borrow<py::dict>(runtime_mod);
  //
  //  assert(runtime_mod_dict.contains("alloc_cpu"));
  //  python_alloc_cpu_fn_ptr = runtime_mod_dict["alloc_cpu"].ptr();
  //
  //  assert(runtime_mod_dict.contains("free_cpu"));
  //  python_free_cpu_fn_ptr = runtime_mod_dict["free_cpu"].ptr();
  //
  //  std::cout << "Py_IsInitialized " << Py_IsInitialized() << "\n";

  atexit(fini);
}

/*********************************************/
/* Here come the malloc function and sisters */
/*********************************************/

void callPythonAllocCpu(void *ptr, size_t size) {
  try {
    py::gil_scoped_acquire guard{};
    if (python_alloc_cpu_fn_ptr) {
      py::function python_alloc_cpu =
          py::reinterpret_borrow<py::function>(python_alloc_cpu_fn_ptr);

      const void *address = static_cast<const void *>(ptr);
      std::stringstream ss;
      ss << address;
      std::string name = ss.str();
      python_alloc_cpu(name, size);
    }
  } catch (py::error_already_set const &pythonErr) {
    std::cout << pythonErr.what();
  }
}

void callPythonFreeCpu(void *ptr) {
  try {
    py::gil_scoped_acquire guard{};
    if (python_free_cpu_fn_ptr) {
      py::function python_free_cpu =
          py::reinterpret_borrow<py::function>(python_free_cpu_fn_ptr);

      const void *address = static_cast<const void *>(ptr);
      std::stringstream ss;
      ss << address;
      std::string name = ss.str();
      python_free_cpu(name);
    }
  } catch (py::error_already_set const &pythonErr) {
    std::cout << pythonErr.what();
  }
}

static void *myalloc_cpu(size_t size) {
  auto ptr = realalloc_cpu(size);
  callPythonAllocCpu(ptr, size);
  return ptr;
}

static void myfree_cpu(void *ptr) {
  callPythonFreeCpu(ptr);
  return realfree_cpu(ptr);
}
