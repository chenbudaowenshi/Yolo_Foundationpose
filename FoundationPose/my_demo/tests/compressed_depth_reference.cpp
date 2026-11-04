// Interoperability check against the installed ROS compressedDepth implementation.
#include <compressed_depth_image_transport/codec.h>
#include <compressed_depth_image_transport/compression_common.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  static_assert(sizeof(compressed_depth_image_transport::ConfigHeader) == 12);
  sensor_msgs::msg::CompressedImage message;
  message.header.stamp.sec = 123456;
  message.header.stamp.nanosec = 987654321;
  message.header.frame_id = "head_d435i_optical_frame";
  message.format = "16UC1; compressedDepth png";
  std::ifstream input(argv[1], std::ios::binary);
  message.data.assign(std::istreambuf_iterator<char>(input), {});
  auto decoded = compressed_depth_image_transport::decodeCompressedDepthImage(message);
  if (!decoded || decoded->encoding != "16UC1" || decoded->width != 256 || decoded->height != 256 ||
      decoded->header != message.header || decoded->data.size() != 65536 * 2) {
    throw std::runtime_error("standard decoder returned wrong image or header");
  }
  for (unsigned i = 0; i < 65536; ++i) {
    uint16_t value;
    std::memcpy(&value, decoded->data.data() + i * 2, sizeof(value));
    if (value != i) throw std::runtime_error("standard decoder changed a depth value");
  }
  // Keep every uint16 value below the reference encoder's range cutoff.
  auto encoded = compressed_depth_image_transport::encodeCompressedDepthImage(*decoded, 65.5359, 100.0, 3);
  if (!encoded || encoded->header != message.header) throw std::runtime_error("standard encoding failed");
  std::ofstream output(argv[2], std::ios::binary);
  output.write(reinterpret_cast<const char *>(encoded->data.data()), encoded->data.size());
  std::cout << "{\"status\":\"PASS\",\"reference_format\":\"" << encoded->format << "\"}\n";
}
