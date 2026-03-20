#pragma once
#include <cstdint>
#include <string_view>

enum class DrawingType : int64_t {
    HandDrawn = 0,
    Digital = 1
};

enum class CaptureMethod : int64_t {
    Photo = 0,
    Scan = 1,
    NA = 2
};

enum class SourceType : int64_t {
    HandDrawnPhoto = 0,
    HandDrawnScan = 1,
    StampPhoto = 2
};

inline std::string_view toString(DrawingType d) {
    return d == DrawingType::HandDrawn ? "hand_drawn" : "digital";
}
inline std::string_view toString(CaptureMethod c) {
    switch (c) {
    case CaptureMethod::Photo: return "photo";
    case CaptureMethod::Scan:  return "scan";
    default:                   return "na";
    }
}
inline std::string_view toString(SourceType s) {
    switch (s) {
    case SourceType::HandDrawnPhoto: return "hand_photo";
    case SourceType::HandDrawnScan:  return "hand_scan";
    case SourceType::StampPhoto:     return "stamp_photo";
    }
    return "unknown";
}
