

namespace tensormorph::cuda{
    class Event{
        private:
            void* handle_ = nullptr;
            int device_index_ = 0;

        public:
            explicit Event(int device_index = 0);

            ~Event() noexcept;

            Event(const Event& )= delete;
            Event& operator=(const Event&) = delete;

            Event(Event&& other) noexcept;
            Event& operator= (Event&& other) noexcept;

            void record(void* stream = nullptr);
            void synchronize() const;

            float elapsed_ms(const Event& end) const;
            void* native_handle() const noexcept;
    };
}