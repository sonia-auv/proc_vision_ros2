// #include <gtest/gtest.h>

// #include <depth_port_manager/ImpactSubsea.hpp>


// TEST(ReadDataCheck, BadStartChar)
// {
//     char data[] = {'6', 'a', '5', '\n'};
//     depth_port_manager::ImpactSubsea device;
//     char buffer[10];
//     ASSERT_EQ(device.ReadDataCheck(
//                   [&](uint8_t* pData, int offset) {
//                       pData[offset] = data[offset];
//                       return 1;
//                   },
//                   buffer, 10),
//               0);
// }

// TEST(ReadDataCheck, NoEndBuffTooSmall)
// {
//     char data1[] = {'$', '1', '2', '3', '4', '5'};

//     depth_port_manager::ImpactSubsea device;
//     char buffer[10];
//     ASSERT_EQ(device.ReadDataCheck(
//                   [&](uint8_t* pData, int offset) {
//                       pData[offset] = data1[offset];
//                       return 1;
//                   },
//                   buffer, 10),
//               -1);
//     char data2[] = {'$', '1', '2', '3', '4', '5', '0', '0', '0', '0', '0', '0', '\n'};
//     ASSERT_EQ(device.ReadDataCheck(
//                   [&](uint8_t* pData, int offset) {
//                       pData[offset] = data2[offset];
//                       return 1;
//                   },
//                   buffer, 10),
//               -1);
// }

// TEST(ReadDataCheck, BadID)
// {
//     std::stringstream data;
//     data << "$"
//          << "3456\n";
//     depth_port_manager::ImpactSubsea device;
//     char buffer[10];
//     ASSERT_EQ(device.ReadDataCheck(
//                   [&](uint8_t* pData, int offset) {
//                       pData[offset] = data.str().c_str()[offset];
//                       return 1;
//                   },
//                   buffer, 10),
//               -2);
// }

// TEST(ReadDataCheck, GoodMessage)
// {
//     std::stringstream data;
//     data << "$";
//     data << depth_port_manager::ImpactSubsea::ID1;
//     data << "3456\n";
//     depth_port_manager::ImpactSubsea device;
//     char buffer[15];
//     ASSERT_EQ(device.ReadDataCheck(
//                   [&](uint8_t* pData, int offset) {
//                       pData[offset] = data.str().c_str()[offset];
//                       return 1;
//                   },
//                   buffer, 15),
//               data.str().size());
// }

// TEST(ParseData, parseID123)
// {
//     // $ISDPT,dddd.ddd,M,ppp.pppp,B,tt.tt,C*xx<CR><LF>
//     depth_port_manager::ImpactSubsea device;
//     depth_port_manager::DepthData output;
//     output.depth = 123.456;
//     output.press = 789.0123;
//     output.temp = 23.45;
//     char buffer[41] = {'$', 'I', 'S', 'D', 'P', 'T', ',', '0', '1', '2', '3', '.',  '4', '5',
//                        '6', ',', 'M', ',', '7', '8', '9', '.', '0', '1', '2', '3',  ',', 'B',
//                        ',', '2', '3', '.', '4', '5', ',', 'C', '*', 'x', 'x', '\n', '\0'};
//     ASSERT_EQ(device.ParseData(buffer), output);
// }

// TEST(Tare, expectedTareCmd)
// {
//     depth_port_manager::ImpactSubsea device;
//     std::string output;
//     device.Tare([&](std::string val) -> ssize_t {
//         output = val;
//         return 0;
//     });
//     ASSERT_EQ(output, "#tare\n");
// }