
namespace tensormorph::cuda {
class Stream {

private:
  // responsible for holding the native cuda handle
  void *handle_;

  int device_index_ = 0;

public:
  explicit Stream(int device_id = 0);

  ~Stream() noexcept;

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  Stream(Stream &&other) noexcept;
  Stream &operator=(Stream &&other) noexcept;

  void *native_handle() const noexcept;
  int device_index() const noexcept;
  void synchronize() const noexcept;
};

} // namespace tensormorph::cuda
